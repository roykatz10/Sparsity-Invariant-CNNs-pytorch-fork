import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from datetime import datetime

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.SparseConvNet import SparseConvNet
from models.ConvNet import ConvNet
device = th.device("cuda" if th.cuda.is_available() else "cpu")


######################################################
# EXPERIMENT PARAMETERS
######################################################

# Experiment Settings
SPARSITY = 0.05
# model = SparseConvNet('Fix').to(device) 
model = ConvNet().to(device)

# Export Settings
RESULT_OUTPUT_FILE = 'ConvNet0.05'
RESULT_OUTPUT_DIR = 'results'

#Training Settings
MAX_EPOCHS = 10
REPEATS = 3

#Optimizer Settings
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-4

criterion = th.nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

######################################################


#append date time to filename --> _YYmmddHHMMSS
now = datetime.now()
RESULT_OUTPUT_FILE += f'_{now.strftime("%Y%m%d%H%M%S")}'
path = f'{RESULT_OUTPUT_DIR}/{RESULT_OUTPUT_FILE}.csv'

data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_set = DataLoader(data, batch_size=128, shuffle=True)

data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_set = DataLoader(data, batch_size=128, shuffle=False)



train_loss, train_acc = [], []
test_loss, test_acc = [], []
epochs = []
repeats = []

for r in range(REPEATS):
    for e in range(MAX_EPOCHS):
        print(f'>>>> Repeat {r} Epoch {e}')
        epochs.append(e)
        repeats.append(r)
        # training
        losses, accuracies = [], []
        for x, y in train_set:
            x, y = x.to(device), y.to(device)

            # mask
            # mask = (x>0).float().to(device)
            mask = (th.rand(x.shape) > SPARSITY).float().to(device)

            mx = model(x, mask, e, device)
            optimizer.zero_grad()
            loss = criterion(mx, y)
            loss.backward()
            optimizer.step() 
            losses.append(loss.item())
            acc = (mx.max(dim=-1)[1] == y).sum().item() / len(y)
            accuracies.append(acc)
        train_loss.append(np.mean(losses))
        train_acc.append(np.mean(acc))

        # testing
        losses, accuracies = [], []
        for x, y in test_set:
            x, y = x.to(device), y.to(device)

            # bruh mask
            # mask = (x>=0).float().to(device)
            mask = (th.rand(x.shape) > SPARSITY).float().to(device)

            mx = model(x, mask, e, device)
            losses.append(criterion(mx, y).item())
            accuracies.append((mx.max(dim=-1)[1] == y).sum().item() / len(y))
        test_loss.append(np.mean(losses))
        test_acc.append(np.mean(accuracies))


data = pd.DataFrame([repeats, epochs, train_acc, test_acc, train_loss, test_loss], 
                    index=['repeat', 'epoch', 'train acc', 'test acc', 'train loss', 'test loss']).transpose()

#epxort data
data.to_csv(path, index=False)

# plt.figure()
# plt.plot(epochs, train_acc)
# plt.plot(epochs, test_acc)
# plt.legend(['Train', 'Test'])
# plt.title('Accuracy')
# plt.xlabel('epoch')
# plt.ylabel('acc')

# plt.figure()
# plt.plot(epochs, train_loss)
# plt.plot(epochs, test_loss)
# plt.legend(['Train', 'Test'])
# plt.title('Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')

# plt.show()