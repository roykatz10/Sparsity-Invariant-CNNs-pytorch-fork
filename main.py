import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from PIL import Image

import transforms as ext_transforms
from models.SparseConvNet import SparseConvNet
from train import Train
from test import Test
from args import get_arguments
import utils

from matplotlib import pyplot as plt
# from data import CamVid as dataset


# Get the arguments
args = get_arguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def load_dataset(dataset):
#     print("\nLoading dataset...\n")

#     print("Selected dataset:", args.dataset)
#     print("Dataset directory:", args.dataset_dir)
#     print("Save directory:", args.save_dir)

#     image_transform = ext_transforms.RandomCrop(336)
#     val_transform = transforms.ToTensor()

#     train_set = dataset(
#         args.dataset_dir,
#         transform=image_transform)
#     train_loader = data.DataLoader(
#         train_set,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.workers)

#     # Load the validation set as tensors
#     val_set = dataset(
#         args.dataset_dir,
#         transform=val_transform,
#         mode='val')
#     val_loader = data.DataLoader(
#         val_set,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.workers)

#     # Load the test set as tensors
#     test_set = dataset(
#         args.dataset_dir,
#         transform=val_transform,
#         mode='test')
#     test_loader = data.DataLoader(
#         test_set,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.workers)

#     return train_loader, val_loader, test_loader


def load_dataset():

    data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    data_train, data_val = random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(data_train, batch_size=128, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=128, shuffle=True)


    data_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(data_test, batch_size=128, shuffle=False)

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
    # if args.resume:
    #     model, optimizer, start_epoch, best_loss = utils.load_checkpoint(
    #         model, optimizer, args.save_dir, args.name)
    #     print("Resuming from model: Start epoch = {0} "
    #           "| Best mean loss = {1:.4f}".format(start_epoch, best_loss))
    # else:
    #     start_epoch = 0
    #     best_loss = 1000

    start_epoch = 0
    best_loss = 1000
    total_epochs = 10

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, device)
    val = Test(model, val_loader, criterion, device)

    train_acc = []
    train_loss = []

    val_acc = []
    val_loss = []
    epochs = []


    for epoch in range(start_epoch, total_epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epochs.append(epoch)

        epoch_loss, epoch_acc = train.run_epoch(lr_updater, epoch, args.print_step)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".format(epoch, epoch_loss))
        
        print(f'>>>> [Epoch: {epoch}] Avg. loss: {epoch_loss:.4f} Avg. acc: {epoch_acc:.4f}')

        if (epoch + 1) % 1 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, acc = val.run_epoch(args.print_step)
            
            val_acc.append(acc)
            val_loss.append(loss)

            # print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".format(epoch, loss))
            
            print(f'>>>> [Epoch: {epoch}] Avg. loss: {loss:.4f} Avg. acc: {acc:.4f}')


            # Save the model if it's the best thus far
            if loss < best_loss:
                print(f"\nBest model thus far is loss: {loss:.4f}, acc: {acc:.4f}.\n")
                best_loss = loss
                
                #TODO: utils.save_checkpoint(model, optimizer, epoch + 1, best_loss, args)


    plt.figure()
    plt.plot(epochs, train_acc)
    plt.plot(epochs, val_acc)
    plt.legend(['Train', 'Validation'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.figure()
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.legend(['Train', 'Validation'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()

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
    # assert os.path.isdir(
    #     args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
    #         args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    # assert os.path.isdir(
    #     args.save_dir), "The directory \"{0}\" doesn't exist.".format(
    #         args.save_dir)

    train_loader, val_loader, test_loader = load_dataset()

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
