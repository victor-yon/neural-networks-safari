from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

from dataset import MNISTDataModule
from network import CNN

AVAIL_GPUS = min(1, torch.cuda.device_count())


def cli_main():
    pl.seed_everything(42, workers=True)

    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CNN.add_model_specific_args(parser)
    args = parser.parse_args()

    # Dataset
    datamodule = MNISTDataModule(batch_size=args.batch_size)

    # Model
    model = CNN(args.learning_rate)

    # Training
    trainer = Trainer(deterministic=True, max_epochs=20, gpus=AVAIL_GPUS)
    trainer.fit(model, datamodule)

    # Testing
    trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    cli_main()
