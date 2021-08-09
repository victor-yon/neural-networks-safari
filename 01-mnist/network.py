from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as f
from torchmetrics.functional import accuracy


class CNN(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3))

        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(f.max_pool2d(self.conv2(x), 2))
        x = f.relu(self.conv3(x))
        x = f.relu(f.max_pool2d(self.conv4(x), 2))
        x = f.relu(f.max_pool2d(self.conv5(x), 2))

        x = x.flatten(start_dim=1)  # Flatten

        x = f.dropout(x, 0.4, training=self.training)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
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
