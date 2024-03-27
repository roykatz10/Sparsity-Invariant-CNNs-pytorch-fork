import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from PIL import Image

import transforms as ext_transforms

from torchvision import transforms,datasets

from models.SparseConvNet import SparseConvNet
from train import Train
from test import Test
from args import get_arguments
import utils
from data import CamVid as dataset
import numpy as np

# Get the arguments
args = get_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(device)


def load_dataset(dataset):
    np.random.seed(42)
    torch.manual_seed(42)

    # In[3]:

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    return train_loader, val_loader, test_loader


def train(train_loader, val_loader):
    print("\nTraining...\n")

    model = SparseConvNet().to(device)
    criterion = nn.MSELoss(reduction='none')

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_loss = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean loss = {1:.4f}".format(start_epoch, best_loss))
    else:
        start_epoch = 0
        best_loss = 1000

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, device)
    val = Test(model, val_loader, criterion, device)
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss = train.run_epoch(lr_updater, args.print_step)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".
              format(epoch, epoch_loss))

        if (epoch + 1) % 1 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".
                  format(epoch, loss))

            # Save the model if it's the best thus far
            if loss < best_loss:
                print("\nBest model thus far. Saving...\n")
                best_loss = loss
                utils.save_checkpoint(model, optimizer, epoch + 1, best_loss,
                                      args)

    return model


def test(model, test_loader):
    print("\nTesting...\n")

    criterion = nn.MSELoss()

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, device)

    print(">>>> Running test dataset")
    loss, (iou, miou) = test.run_epoch(args.print_step)
    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))


# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the dataset directory doesn't exist

    # Fail fast if the saving directory doesn't exist
    # assert os.path.isdir(
    #     args.save_dir), "The directory \"{0}\" doesn't exist.".format(
    #         args.save_dir)

    train_loader, val_loader, test_loader = load_dataset(dataset)

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new SparseConvNet model
            model = SparseConvNet().to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the SparseConvNet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]

        test(model, test_loader)
