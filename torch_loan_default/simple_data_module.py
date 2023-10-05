import requests
import os
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np


class RandomPolynomialDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_samples=1000,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def setup(self, stage=None):
        # Generate random x values
        x = torch.rand(self.num_samples) * 10 - 5  # Random values in the range [-5, 5]

        # Calculate corresponding y values using the polynomial function
        y = x**3 + 7 * x + 8

        # Split the data into training, validation, and test sets
        train_size = int(self.train_ratio * self.num_samples)
        val_size = int(self.val_ratio * self.num_samples)
        test_size = self.num_samples - train_size - val_size

        self.x_train, self.x_val, self.x_test = np.split(
            x.numpy(), [train_size, train_size + val_size]
        )
        self.y_train, self.y_val, self.y_test = np.split(
            y.numpy(), [train_size, train_size + val_size]
        )

    def train_dataloader(self):
        train_dataset = TensorDataset(
            torch.Tensor(self.x_train), torch.Tensor(self.y_train)
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = TensorDataset(torch.Tensor(self.x_val), torch.Tensor(self.y_val))
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = TensorDataset(
            torch.Tensor(self.x_test), torch.Tensor(self.y_test)
        )
        return DataLoader(test_dataset, batch_size=self.batch_size)


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
import numpy as np


class EvenOddDataModule(pl.LightningDataModule):
    def __init__(
        self,
        width,
        batch_size=32,
        num_samples=10000,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    ):
        super().__init__()
        self.width = width
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def setup(self, stage=None):
        # Generate random x values
        x = torch.floor(
            torch.rand(self.num_samples, self.width) * 10 - 5
        )  # Random values in the range [-5, 5] for each feature

        # Calculate corresponding y values using the polynomial function
        y = torch.prod(x, dim=1).long() % 2  # 1 if even, 0 if odd
        y = y.view(-1, 1)

        # Split the data into training, validation, and test sets
        train_size = int(self.train_ratio * self.num_samples)
        val_size = int(self.val_ratio * self.num_samples)
        test_size = self.num_samples - train_size - val_size

        self.x_train, self.x_val, self.x_test = np.split(
            x.numpy(), [train_size, train_size + val_size]
        )
        self.y_train, self.y_val, self.y_test = np.split(
            y.numpy(), [train_size, train_size + val_size]
        )

    def train_dataloader(self):
        train_dataset = TensorDataset(
            torch.Tensor(self.x_train), torch.Tensor(self.y_train).view(-1, 1)
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = TensorDataset(
            torch.Tensor(self.x_val), torch.Tensor(self.y_val).view(-1, 1)
        )
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = TensorDataset(
            torch.Tensor(self.x_test), torch.Tensor(self.y_test).view(-1, 1)
        )
        return DataLoader(test_dataset, batch_size=self.batch_size)
