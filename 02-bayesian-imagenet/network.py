from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as f
from torchmetrics.functional import accuracy


class BayesCNN(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)  # Flatten
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return f.log_softmax(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = f.cross_entropy(out, y)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        out = self(x)
        loss = f.nll_loss(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
