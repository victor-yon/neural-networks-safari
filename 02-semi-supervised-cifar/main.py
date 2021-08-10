from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

from dataset import CIFARDataModule
from network import BayesCNN

AVAIL_GPUS = min(1, torch.cuda.device_count())


def cli_main():
    pl.seed_everything(42, workers=True)

    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BayesCNN.add_model_specific_args(parser)
    args = parser.parse_args()

    # Dataset
    datamodule = CIFARDataModule(batch_size=args.batch_size, augmentation=False, validation_split=0.1)
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.plot_sample(20)

    # Model
    model = BayesCNN(args.learning_rate)

    # Training
    trainer = Trainer(deterministic=True, max_epochs=20, gpus=AVAIL_GPUS)
    trainer.fit(model, datamodule)

    # Testing
    trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    cli_main()
