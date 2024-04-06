import torch.nn as nn
import torch
from typing import Literal

class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 normalizer_type:Literal['Original', 'Fix', 'None']):
        super().__init__()

        self.normalizer_type = normalizer_type

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
        # normalizer = (self.kernel_size * self.kernel_size)/ (self.sparsity(mask)+1e-8)
        # normalizer = (torch.zeros((normalizer.shape)) + (1 / self.kernel_size)).to(device)
        # normalizer = (torch.ones(normalizer.shape)).to(device)

        if self.normalizer_type == 'Original':
            normalizer = 1 / (self.sparsity(mask)+1e-8)
        elif self.normalizer_type == 'Fix':
            normalizer = (self.kernel_size * self.kernel_size)/ (self.sparsity(mask)+1e-8)
        elif self.normalizer_type == 'None':
            normalizer = torch.ones(x.shape).to(device)
        else:
            raise Exception(f'Type not implemented: {self.normalizer_type}')


        # normalizer.to(device)


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

    def __init__(self, architecture:Literal['Original', 'New'] ,normalizer_type:Literal['Original', 'Fix', 'None']):
        super().__init__()

        if architecture == 'Original':
            self.layers = nn.ParameterList([SparseConv(1, 16, 11, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 7, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 5, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 3, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 3, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 1, 1, normalizer_type),      # ( 1, 28, 28)
                                            nn.Flatten(),                               # ( 784)
                                            nn.Linear(784, 10)                          # ( 10)
                                            ])
            
        elif architecture == 'New':
            self.layers = nn.ParameterList([SparseConv(1, 16, 11, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 7, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 5, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 3, normalizer_type),     # (16, 28, 28)
                                            SparseConv(16, 16, 3, normalizer_type),     # (16, 28, 28)
                                            nn.Flatten(),                               # ( 12544)
                                            nn.Linear(12544, 784),                      # ( 784)
                                            nn.ReLU(),                                  # ( 784)
                                            nn.Linear(784, 10)                          # ( 10)
                                            ])
            
        else:
            raise Exception(f'Architecture not implemented: {architecture}')

        # self.SparseLayer1 = SparseConv(1, 16, 11, normalizer_type)
        # self.SparseLayer2 = SparseConv(16, 16, 7, normalizer_type)
        # self.SparseLayer3 = SparseConv(16, 16, 5, normalizer_type)
        # self.SparseLayer4 = SparseConv(16, 16, 3, normalizer_type)
        # self.SparseLayer5 = SparseConv(16, 16, 3, normalizer_type)
        # self.SparseLayer6 = SparseConv(16, 1, 1, normalizer_type)

        # self.flatten = nn.Flatten()
        # self.linear0 = nn.Linear(12544, 784)
        # self.activation0 = nn.ReLU()
        # self.linear1 = nn.Linear(784, 10)

    def forward(self, x, mask, epoch_nr, device):
                                             # ( 1, 28, 28)
        # x, mask = self.SparseLayer1(x, mask, epoch_nr, device) # (16, 28, 28)
        # x, mask = self.SparseLayer2(x, mask, epoch_nr, device) # (16, 28, 28)
        # x, mask = self.SparseLayer3(x, mask, epoch_nr, device) # (16, 28, 28)
        # x, mask = self.SparseLayer4(x, mask, epoch_nr, device) # (16, 28, 28)
        # x, mask = self.SparseLayer5(x, mask, epoch_nr, device) # (16, 28, 28)
        # x, mask = self.SparseLayer6(x, mask, epoch_nr, device) # ( 1, 28, 28)

        # x = self.flatten(x) # (1 x 28 x 28 = 784)
        # # x = self.linear0(x)
        # # x = self.activation0(x)
        # x = self.linear1(x) # (10)




        for layer in self.layers:
            
            if isinstance(layer, SparseConv):
                x, mask = layer(x, mask, epoch_nr, device)
            
            else:
                x = layer(x)

        return x
