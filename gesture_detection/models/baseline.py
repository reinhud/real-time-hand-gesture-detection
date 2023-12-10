from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


class Baseline(L.LightningModule):

    def __init__(self, num_classes: int, lr: float = 0.01):
        super().__init__()
        self.lr = lr
        self.num_features = 160
        self.num_classes = num_classes

        self.backbone = torchvision.models.mobilenet_v3_large()
        self.backbone.features = nn.Sequential(
            *[module for idx, module in enumerate(self.backbone.features.children()) if idx < 14]
        )
        self.backbone.classifier = nn.Identity()

        self.sequence_model = nn.LSTM(self.num_features, 100, 4, dropout=0.05)
        self.linear1 = nn.Linear(100, self.num_classes)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape

        ret = torch.zeros((batch_size, time_steps, self.num_classes))
        t = 0
        y = self.backbone(x[:, t])
        out, (hn, cn) = self.sequence_model(y.unsqueeze(1))
        ret[:, t] = self.linear1(out.squeeze(1))
        for t in range(1, time_steps):
            y = self.backbone(x[:, t])
            out, (hn, cn) = self.sequence_model(y.unsqueeze(1), (hn, cn))
            ret[:, t] = self.linear1(out.squeeze(1))
        return ret

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        batch_size, sample_length = inputs.shape[:2]
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes

        # since targets contains only one int per time series, repeat it for the number of samples in a sequence
        targets = torch.repeat_interleave(targets, sample_length)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        inputs, targets = batch
        batch_size, sample_length = inputs.shape[:2]
        outputs = self(inputs).flatten(0, 1)  # [batch_size * sample_length, num_classes

        # since targets contains only one int per time series, repeat it for the number of samples in a sequence
        targets = torch.repeat_interleave(targets, sample_length)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.Adam([
                #  {"params": self.backbone.parameters(), "lr": 0.0001},
                {"params": self.sequence_model.parameters()},
                {"params": self.linear1.parameters()}
            ], lr=self.lr)
        return opt

def main():
    model = Baseline()
    out = model(torch.rand((1, 5, 3, 160, 160)))
    preds = F.softmax(out, dim=-1).argmax(dim=-1).squeeze(0).detach().numpy()
    print(f"predicted classes: {preds}")


if __name__ == "__main__":
    main()
