import torch as th
import numpy as np

import matplotlib.pyplot as plt


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = th.device("cuda" if th.cuda.is_available() else "cpu")

data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_set = DataLoader(data, batch_size=128, shuffle=True)

data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_set = DataLoader(data, batch_size=128, shuffle=False)




# MNIST images are (1, 28, 28) dimensional
model = th.nn.Sequential(th.nn.Conv2d(1, 16, 11),                   # (16, 18, 18)
                         th.nn.ReLU(),                              # (16, 18, 18)
                         th.nn.Conv2d(16, 16, 7),                   # (16, 12, 12)
                         th.nn.ReLU(),                              # (16, 12, 12)
                         th.nn.Conv2d(16, 16, 5),                   # (16,  8,  8)
                         th.nn.ReLU(),                              # (16,  8,  8)
                         th.nn.Conv2d(16, 16, 3),                   # (16,  6,  6)
                         th.nn.ReLU(),                              # (16,  6,  6)
                         th.nn.Conv2d(16, 16, 3),                   # (16,  4,  4)
                         th.nn.ReLU(),                              # (16,  4,  4)
                        #  th.nn.Conv2d(16, 1, 1),                  # ( 1,  4,  4)
                        #  th.nn.ReLU(),                            # ( 1,  4,  4)
                         th.nn.Flatten(),                           # (256)
                         th.nn.Linear(256, 128),   # (128)
                         th.nn.ReLU(),             # (128)
                         th.nn.Linear(128, 32),    # ( 32)
                         th.nn.ReLU(),             # ( 32)
                         th.nn.Linear(32, 10)      # ( 10)
                        ).to(device)


# model = th.nn.Sequential(th.nn.Conv2d(1, 8, 5),    # ( 8, 24, 24)
#                          th.nn.ReLU(),             # ( 8, 24, 24)
#                          th.nn.MaxPool2d(2, 2),    # ( 8, 12, 12)
#                          th.nn.Conv2d(8, 16, 5),   # (16,  8,  8)
#                          th.nn.ReLU(),             # (16,  8,  8)
#                          th.nn.MaxPool2d(2, 2),    # (16,  4,  4)
#                          th.nn.Flatten(),          # (16 * 4 * 4)
#                          th.nn.Linear(256, 128),   # (128)
#                          th.nn.ReLU(),             # (128)
#                          th.nn.Linear(128, 32),    # ( 32)
#                          th.nn.ReLU(),             # ( 32)
#                          th.nn.Linear(32, 10)      # ( 10)
#                         ).to(device)  


# model = th.nn.Sequential(th.nn.Conv2d(1, 8, 5),    # ( 8, 24, 24)
#                          th.nn.ReLU(),             # ( 8, 24, 24)
#                          th.nn.Conv2d(8, 16, 5),   # (16, 20, 20)
#                          th.nn.ReLU(),             # (16, 20, 20)
#                          th.nn.Flatten(),          # (16 * 20 * 20)
#                          th.nn.Linear(6400, 128),   # (128)
#                          th.nn.ReLU(),             # (128)
#                          th.nn.Linear(128, 32),    # ( 32)
#                          th.nn.ReLU(),             # ( 32)
#                          th.nn.Linear(32, 10)      # ( 10)
#                         ).to(device)     


criterion = th.nn.CrossEntropyLoss()
# optimizer = th.optim.RMSprop(model.parameters(), lr=0.0005)

optimizer = th.optim.Adam(model.parameters(),
                          lr=1e-4,
                          weight_decay=2e-4)

max_epochs = 10

train_loss, train_acc = [], []
test_loss, test_acc = [], []
epochs = list(range(max_epochs))

for e in range(max_epochs):
    # training
    losses, accuracies = [], []
    for x, y in train_set:
        x, y = x.to(device), y.to(device)

        mx = model(x)
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

        mx = model(x)
        losses.append(criterion(mx, y).item())
        accuracies.append((mx.max(dim=-1)[1] == y).sum().item() / len(y))
    test_loss.append(np.mean(losses))
    test_acc.append(np.mean(accuracies))


plt.figure()
plt.plot(epochs, train_acc)
plt.plot(epochs, test_acc)
plt.legend(['Train', 'Test'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')

plt.figure()
plt.plot(epochs, train_loss)
plt.plot(epochs, test_loss)
plt.legend(['Train', 'Test'])
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()