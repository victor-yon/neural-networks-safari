import os
from math import ceil, floor
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

PATH_DATASETS = os.environ.get('PATH_DATASETS', '../cache/datasets')
NUM_WORKERS = int(os.cpu_count() / 2)


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, augmentation: bool = True, validation_split: float = 0):
        super().__init__()
        self.data_dir = PATH_DATASETS
        self.batch_size = batch_size
        self.validation_split = validation_split

        if augmentation:
            tr = [
                transforms.RandomRotation(10)
            ]
        else:
            tr = []

        tr.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform = transforms.Compose(tr)

        # Nothing loaded yet
        self.cifar_train = self.cifar_val = self.cifar_test = None

    def prepare_data(self):
        # Download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            nb_train_item = len(cifar_full)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [
                floor(nb_train_item * (1 - self.validation_split)),
                ceil(nb_train_item * self.validation_split)])

            # Save data dimension
            self.dims = tuple(self.cifar_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

            # Save data dimension
            self.dims = tuple(self.cifar_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=NUM_WORKERS)
