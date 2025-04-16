"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-04-27 22:53:54
LastEditTime: 2023-10-22 17:31:05
Contact: cgjcuhk@gmail.com
Description: debug datasets
"""


from pl_bolts.datasets import DummyDataset
# from torchvision.datasets import FakeData
from torch.utils.data import DataLoader


class DummyDataModule:
    def __init__(self, batch_size: int = 1, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)
        # self.data_train = FakeData(size=1, image_size=(1, 28, 28))
        self.data_train = DummyDataset((1, 14, 14), (1,), num_samples=1)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = DummyDataModule()
