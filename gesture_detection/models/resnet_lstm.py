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


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create residual block with two conv layers.

        Parameters:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
            - stride (int): Stride for first convolution.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual = nn.Identity() if (in_channels == out_channels and stride == 1) else nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################

    def forward(self, x):
        """
        Compute the forward pass through the residual block.

        Parameters:
            - x (torch.Tensor): Input.

        Returns:
            - out (torch.tensor): Output.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))

        out = out + self.residual(x)
        out = F.relu(out)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


class ResNetLSTM(L.LightningModule):

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

        residual_config = [16, 32, 32]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=residual_config[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(residual_config[0])

        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(
                in_channels=residual_config[idx],
                out_channels=residual_config[idx + 1],
                stride=2
            ) for idx in range(len(residual_config) - 1)
        ])

        avg_pool_size = 2
        self.avg_pool = nn.AdaptiveAvgPool2d(2)
        self.flatten = nn.Flatten()
        self.classifier = nn.LSTM(avg_pool_size ** 2 * residual_config[-1], 14, batch_first=True)

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.residual_blocks(out)

        out = self.avg_pool(out).flatten(1, 2, 3)
        out = out.unflatten(0, (batch_size, time_steps))
        out, (hn, cn) = self.classifier(out)
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
        opt = torch.optim.Adam([
            {"params": self.backbone.parameters(), "lr": self.backbone_lr},
            {"params": self.sequence_model.parameters()},
        ], lr=self.lr, weight_decay=self.weight_decay)
        return opt


def main():
    model = ResNetLSTM()

    def capacity(module):
        num_param = sum(p.numel() for p in module.parameters() if p.requires_grad)

        return num_param

    print(capacity(model))
    out = model(torch.rand((2, 5, 3, 120, 120)))
    preds = F.softmax(out, dim=-1).argmax(dim=-1).squeeze(0).detach().numpy()
    print(f"predicted classes: {preds}")


if __name__ == "__main__":
    main()
