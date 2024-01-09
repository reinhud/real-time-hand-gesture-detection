from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import F1Score, Accuracy, ConfusionMatrix


class Baseline(L.LightningModule):

    def __init__(
            self,
            num_classes: int,
            lr: float = 0.01, backbone_lr: float = 0.001,
            weight_decay: float = 0.0,
            loss_weight: list[float] | None = None
    ):
        super().__init__()
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.loss_weight = torch.tensor(loss_weight) if loss_weight is not None else None
        self.num_features = 960
        self.num_classes = num_classes

        self.backbone = torchvision.models.mobilenet_v3_large()
        #self.backbone.features = nn.Sequential(
        #    *[module for idx, module in enumerate(self.backbone.features.children()) if idx < 14]
        #)
        self.backbone.classifier = nn.Identity()
        self.linear1 = nn.Linear(self.num_features, self.num_classes)

        self.metric_config = {
            "acc": (Accuracy, {"task": "multiclass", "num_classes": num_classes}),
            "f1": (F1Score, {"task": "multiclass", "num_classes": num_classes}),
            #"cm": (ConfusionMatrix, {"task": "multiclass", "num_classes": num_classes})
        }

        for stage in ["train", "valid", "test"]:
            for metric in self.metric_config.keys():
                metric_cls, metric_params = self.metric_config[metric]
                setattr(self, f"{metric}_{stage}", metric_cls(**metric_params))


    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.flatten(0, 1)
        x = self.backbone(x)
        x = self.linear1(x)
        return x

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        batch_size, sample_length = inputs.shape[:2]
        outputs = self(inputs)  # [batch_size * sample_length, num_classes

        targets = targets.flatten(0, 1)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        stage = "train"
        for metric in self.metric_config.keys():
            metric_attr = getattr(self, f"{metric}_{stage}")
            metric_attr(outputs, targets)
            self.log(f"{stage}_{metric}", metric_attr, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        batch_size, sample_length = inputs.shape[:2]
        outputs = self(inputs)  # [batch_size * sample_length, num_classes

        targets = targets.flatten(0, 1)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        stage = "valid"
        for metric in self.metric_config.keys():
            metric_attr = getattr(self, f"{metric}_{stage}")
            metric_attr(outputs, targets)
            self.log(f"{stage}_{metric}", metric_attr, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        batch_size, sample_length = inputs.shape[:2]
        outputs = self(inputs)  # [batch_size * sample_length, num_classes

        targets = targets.flatten(0, 1)
        loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.loss_weight)

        stage = "test"
        for metric in self.metric_config.keys():
            metric_attr = getattr(self, f"{metric}_{stage}")
            metric_attr(outputs, targets)
            self.log(f"{stage}_{metric}", metric_attr, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.Adam([
                {"params": self.backbone.parameters(), "lr": self.backbone_lr},
                {"params": self.linear1.parameters()}
            ], lr=self.lr, weight_decay=self.weight_decay)
        return opt

def main():
    model = Baseline()
    out = model(torch.rand((1, 5, 3, 160, 160)))
    preds = F.softmax(out, dim=-1).argmax(dim=-1).squeeze(0).detach().numpy()
    print(f"predicted classes: {preds}")


if __name__ == "__main__":
    main()
