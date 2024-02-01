from typing import Any, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensorboardX import SummaryWriter
from torchmetrics import F1Score, Accuracy, ConfusionMatrix, MeanMetric
from torchmetrics.classification import MulticlassConfusionMatrix

from gesture_detection.utility.SequenceMetric import SequenceMetric
from gesture_detection.utility.plot_confusion_matrix import plot_confusion_matrix


class SummaryWriterLogger:

    def __init__(self):
        self._writer: SummaryWriter = None

    @property
    def writer(self):
        return self._writer

    @writer.setter
    def writer(self, value):
        self._writer = value


    def add_scalar(self, metric_name, value, step):
        if self._writer is None:
            print("Writer is not set yet")
            return
        self._writer.add_scalar(
            metric_name,
            value,
            step,
        )
        self._writer.flush()

    def add_figure(self, tag, fig, step):
        if self._writer is None:
            print("Writer is not set yet")
            return
        self._writer.add_figure(
            tag, fig, step
        )
        self._writer.flush()

    def add_hparams(self, hparams):
        if self._writer is None:
            print("Writer is not set yet")
            return
        self._writer.add_hparams(hparams, {})
        self._writer.add_text(
            "hparams", str(hparams)
        )
        self._writer.flush()


class LSTM(L.LightningModule):

    def __init__(
            self,
            num_classes: int,
            lr: float = 0.01, backbone_lr: float = 0.001,
            weight_decay: float = 0.0,
            loss_weight: list[float] | None = None,
            sample_length: int = 32,
            label_smoothing: float = 0.0,
            small: bool = True
    ):
        super().__init__()
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.register_buffer("loss_weight",
                             torch.tensor(loss_weight) if loss_weight is not None else torch.ones(num_classes)
                             )
        self.small = small
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        # self.save_hyperparameters()
        self.summary_writer = SummaryWriterLogger()

        if self.small:
            self.num_features = 576  # MBv3-S
            self.backbone = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            self.num_features = 960  # MBv3-L
            self.backbone = torchvision.models.mobilenet_v3_large(torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.sequence_model = nn.LSTM(self.num_features, 128, 1, batch_first=True)
        self.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
        )

        self.metric_config = {
            "acc": (Accuracy, {"task": "multiclass", "num_classes": num_classes, "average": "macro"}),
            "f1": (F1Score, {"task": "multiclass", "num_classes": num_classes, "average": "macro"}),
            "sc": (SequenceMetric, {"num_steps": sample_length}),
            "cm": (ConfusionMatrix, {"task": "multiclass", "num_classes": num_classes})
        }

        for stage in ["train", "valid", "test"]:
            for metric in self.metric_config.keys():
                metric_cls, metric_params = self.metric_config[metric]
                setattr(self, f"{metric}_{stage}", metric_cls(**metric_params))
        self.train_loss_epoch = MeanMetric()
        self.valid_loss_epoch = MeanMetric()

    def log_stage(self, stage: str, outputs: torch.Tensor, targets: torch.Tensor, step):
        for metric in self.metric_config.keys():
            metric_attr = getattr(self, f"{metric}_{stage}")
            metric_attr.update(outputs, targets)

    def on_fit_start(self) -> None:
        self.summary_writer.writer = self.logger.experiment
        self.summary_writer.add_hparams({
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "weight_decay": self.weight_decay,
            "small": self.small,
            "label_smoothing": self.label_smoothing
        })

    def on_test_start(self) -> None:
        self.summary_writer.writer = self.logger.experiment

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.flatten(0, 1)
        x = self.backbone(x)
        x = x.unflatten(0, (batch_size, time_steps))
        x, (hn, cn) = self.sequence_model(x)
        out = self.linear(x.flatten(0, 1)).unflatten(0, (batch_size, time_steps))
        return out

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes
        targets = targets.flatten(0, 1)

        loss = torch.nn.functional.cross_entropy(
            outputs, targets, weight=self.loss_weight, label_smoothing=self.label_smoothing
        )

        # self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.summary_writer.add_scalar(
            "train_loss_step",
            loss,
            self.global_step,
        )
        self.train_loss_epoch.update(loss)
        self.log_stage("train", outputs, targets, self.global_step)
        return loss

    def metric_reset(self, stage: str, step):
        for metric in self.metric_config.keys():
            metric_attr = getattr(self, f"{metric}_{stage}")

            if isinstance(metric_attr, SequenceMetric):
                fig, ax = metric_attr.plot()
                self.summary_writer.add_figure(f"{stage}_{metric}", fig, step)
            elif isinstance(metric_attr, MulticlassConfusionMatrix):
                fig, ax = plot_confusion_matrix(metric_attr.compute())
                self.summary_writer.add_figure(f"{stage}_{metric}", fig, step)
            else:
                self.summary_writer.add_scalar(
                    f"{stage}_{metric}",
                    metric_attr.compute(),
                    step,
                )
            metric_attr.reset()
        if hasattr(self, f"{stage}_loss_epoch"):
            loss_fun = getattr(self, f"{stage}_loss_epoch")
            self.summary_writer.add_scalar(
                f"{stage}_loss_epoch",
                loss_fun.compute(),
                step,
            )
            loss_fun.reset()

    def on_train_epoch_end(self) -> None:
        self.metric_reset("train", self.global_step)

    def on_validation_epoch_end(self) -> None:
        self.metric_reset("valid", (self.current_epoch + 1) * self.trainer.num_val_batches[0])

    def on_test_epoch_end(self) -> None:
        self.metric_reset("test", self.global_step, (self.current_epoch + 1) * self.trainer.num_test_batches[0])

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes
        targets = targets.flatten(0, 1)

        loss = torch.nn.functional.cross_entropy(
            outputs, targets, weight=self.loss_weight, label_smoothing=self.label_smoothing
        )

        # self.log("valid_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.summary_writer.add_scalar(
            "valid_loss_step",
            loss,
            self.current_epoch * self.trainer.num_val_batches[0] + batch_idx,
        )
        self.valid_loss_epoch.update(loss)
        self.log_stage("valid", outputs, targets, self.current_epoch * self.trainer.num_val_batches[0] + batch_idx)
        return loss

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes
        targets = targets.flatten(0, 1)

        loss = torch.nn.functional.cross_entropy(
            outputs, targets, weight=self.loss_weight, label_smoothing=self.label_smoothing
        )
        self.log_stage("test", outputs, targets, self.current_epoch * self.trainer.num_test_batches[0] + batch_idx)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt


def main():
    model = LSTM(14)
    out = model(torch.rand((2, 5, 3, 160, 160)))
    preds = F.softmax(out, dim=-1).argmax(dim=-1).squeeze(0).detach().numpy()
    print(f"predicted classes: {preds}")


if __name__ == "__main__":
    main()
