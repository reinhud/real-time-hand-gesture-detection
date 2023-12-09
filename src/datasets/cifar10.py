import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from src.config.base_config import get_base_config
from src.utility.find_num_worker_default import find_num_worker_default

base_config = get_base_config()


class CIFAR10DataModule(L.LightningDataModule):
    name = "CIFAR10"

    def __init__(
        self,
        batch_size: int = 32,
        data_dir: str = f"{base_config.DATA_DIR}/cifar10",
        num_workers: int | None = None,
        pin_memory: bool = base_config.USE_GPU,
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = find_num_worker_default(num_workers)
        self.pin_memory = pin_memory
        self.seed = seed
        self.train_ratio = train_ratio
        self.save_hyperparameters()

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
            train_size = int(50000 * self.train_ratio)
            val_size = 50000 - train_size
            self.cifar10_train, self.cifar10_val = random_split(
                dataset=cifar10_full,
                lengths=[train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=transform)

        if stage == "predict" or stage is None:
            self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar10_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
      