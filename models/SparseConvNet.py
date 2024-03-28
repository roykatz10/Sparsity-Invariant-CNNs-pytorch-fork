import torch.nn as nn
import torch


class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        padding = kernel_size//2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)


        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        self.kernel_size = kernel_size
        self.count = 0
        self.epoch = 0

        

    def forward(self, x, mask, epoch_nr, device):
        x = x*mask
        x = self.conv(x)
        # normalizer = 1 / (self.sparsity(mask)+1e-8)
        normalizer = (self.kernel_size * self.kernel_size)/ (self.sparsity(mask)+1e-8)
        # normalizer = (torch.zeros((normalizer.shape)) + (1 / self.kernel_size)).to(device)
        # normalizer = (torch.ones(normalizer.shape)).to(device)

        if epoch_nr > self.epoch:
            self.epoch = epoch_nr
            count = 0

        if False:
        # if self.count < 1 and epoch_nr > 0:
            print(torch.sum(x))

            self.count = 1
            print('x', x)
            print('x* normalizer1', x * normalizer1)
            print('normalizer1', normalizer1)
            # print(self.kernel_size)
            print('xsum', torch.sum(x))
            print('x*n1sum', torch.sum(x* normalizer1))
            print('normalizer1', normalizer1)
        x = (x  * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        x = self.relu(x)
        
        mask = self.max_pool(mask)

        return x, mask



class SparseConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.SparseLayer1 = SparseConv(1, 16, 11)
        self.SparseLayer2 = SparseConv(16, 16, 7)
        self.SparseLayer3 = SparseConv(16, 16, 5)
        self.SparseLayer4 = SparseConv(16, 16, 3)
        self.SparseLayer5 = SparseConv(16, 16, 3)
        self.SparseLayer6 = SparseConv(16, 1, 1)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 10)

    def forward(self, x, mask, epoch_nr, device):
                                             # ( 1, 28, 28)
        x, mask = self.SparseLayer1(x, mask, epoch_nr, device) # (16, 28, 28)
        x, mask = self.SparseLayer2(x, mask, epoch_nr, device) # (16, 28, 28)
        x, mask = self.SparseLayer3(x, mask, epoch_nr, device) # (16, 28, 28)
        x, mask = self.SparseLayer4(x, mask, epoch_nr, device) # (16, 28, 28)
        x, mask = self.SparseLayer5(x, mask, epoch_nr, device) # (16, 28, 28)
        x, mask = self.SparseLayer6(x, mask, epoch_nr, device) # ( 1, 28, 28)

        x = self.flatten(x) # (1 x 28 x 28 = 784)
        x = self.linear1(x) # (10)

        return x
