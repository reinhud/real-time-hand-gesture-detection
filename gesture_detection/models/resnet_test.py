import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models


class LResNetTest(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        # self.save_hyperparameters()

        # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)
        self.log("train_loss", float(loss))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)
        acc = (outputs.argmax(1) == targets).float().mean()
        self.log("val_loss", float(loss))
        self.log("val_acc", float(acc))
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
