from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

from network import CNN


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CNN.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST('../cache/datasets', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('../cache/datasets', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = CNN(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
