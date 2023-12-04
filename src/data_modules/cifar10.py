import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64, data_dir: str = "./data/cifar10"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ]
        )

        # split dataset
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=transform)
            self.cifar10_train, self.cifar10_val = random_split(
                cifar10_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=transform)

        if stage == "predict" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=8)
