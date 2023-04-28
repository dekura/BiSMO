"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-04-27 22:53:54
LastEditTime: 2023-04-28 02:52:43
Contact: cgjcuhk@gmail.com
Description: debug datasets
"""

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from pl_bolts.datasets import DummyDataset
from torch.utils.data import DataLoader


class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 1, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train = DummyDataset((1, 28, 28), (1,), num_samples=1)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = DummyDataModule()
