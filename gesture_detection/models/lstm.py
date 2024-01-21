from typing import Any, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import F1Score, Accuracy, ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix

from gesture_detection.utility.SequenceMetric import SequenceMetric
from gesture_detection.utility.plot_confusion_matrix import plot_confusion_matrix


class LSTM(L.LightningModule):

    def __init__(
            self,
            num_classes: int,
            lr: float = 0.01, backbone_lr: float = 0.001,
            weight_decay: float = 0.0,
            loss_weight: list[float] | None = None,
            sample_length: int = 32,
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
        self.save_hyperparameters()

        if self.small:
            self.num_features = 576  # MBv3-S
            self.backbone = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
            self.backbone.classifier = nn.Identity()
            self.sequence_model = nn.LSTM(self.num_features, self.num_classes, 1, batch_first=True)
        else:
            self.num_features = 960  # MBv3-L
            self.backbone = torchvision.models.mobilenet_v3_large(torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
            self.backbone.classifier = nn.Identity()
            self.sequence_model = nn.LSTM(self.num_features, self.num_classes, 1, batch_first=True)

        self.metric_config = {
            "acc": (Accuracy, {"task": "multiclass", "num_classes": num_classes}),
            "f1": (F1Score, {"task": "multiclass", "num_classes": num_classes}),
            "sc": (SequenceMetric, {"num_steps": sample_length}),
            "cm": (ConfusionMatrix, {"task": "multiclass", "num_classes": num_classes})
        }

        for stage in ["train", "valid", "test"]:
            for metric in self.metric_config.keys():
                metric_cls, metric_params = self.metric_config[metric]
                setattr(self, f"{metric}_{stage}", metric_cls(**metric_params))

    def log_stage(self, stage: str, outputs: torch.Tensor, targets: torch.Tensor):
        for metric in self.metric_config.keys():
            metric_attr = getattr(self, f"{metric}_{stage}")
            metric_attr.update(outputs, targets)

            if not isinstance(metric_attr, Union[SequenceMetric, MulticlassConfusionMatrix]):
                self.log(f"{stage}_{metric}", metric_attr, on_step=False, on_epoch=True)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.flatten(0, 1)
        x = self.backbone(x)
        x = x.unflatten(0, (batch_size, time_steps))
        out, (hn, cn) = self.sequence_model(x)
        return out

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes
        targets = targets.flatten(0, 1)

        loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.loss_weight)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log_stage("train", outputs, targets)
        return loss

    def metric_reset(self, stage: str):
        for metric in self.metric_config.keys():
            metric_attr = getattr(self, f"{metric}_{stage}")

            if isinstance(metric_attr, SequenceMetric):
                fig, ax = metric_attr.plot()
                self.logger.experiment.add_figure(f"{stage}_{metric}", fig, self.global_step)
            elif isinstance(metric_attr, MulticlassConfusionMatrix):
                fig, ax = plot_confusion_matrix(metric_attr.compute())
                self.logger.experiment.add_figure(f"{stage}_{metric}", fig, self.global_step)
            metric_attr.reset()

    def on_train_epoch_end(self) -> None:
        self.metric_reset("train")

    def on_validation_epoch_end(self) -> None:
        self.metric_reset("valid")

    def on_test_epoch_end(self) -> None:
        self.metric_reset("test")

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes
        targets = targets.flatten(0, 1)

        loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.loss_weight)

        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        self.log_stage("valid", outputs, targets)
        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes
        targets = targets.flatten(0, 1)

        loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.loss_weight)

        self.log_stage("test", outputs, targets)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.Adam([
                {"params": self.backbone.parameters(), "lr": self.backbone_lr},
                {"params": self.sequence_model.parameters()},
            ], lr=self.lr, weight_decay=self.weight_decay)
        return opt

def main():
    model = LSTM(14)
    out = model(torch.rand((2, 5, 3, 160, 160)))
    preds = F.softmax(out, dim=-1).argmax(dim=-1).squeeze(0).detach().numpy()
    print(f"predicted classes: {preds}")


if __name__ == "__main__":
    main()
