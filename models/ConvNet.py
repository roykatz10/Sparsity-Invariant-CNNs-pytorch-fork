import torch.nn as nn
import torch
from typing import Literal



class ConvNet(nn.Module):

    def __init__(self, architecture:Literal['Original', 'New']):
        super().__init__()

        if architecture == 'Original':
            self.layers = nn.ParameterList([
                        nn.Conv2d(1, 16, 11, padding=11//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 7, padding= 7//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 5, padding= 5//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 3, padding= 3//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 3, padding= 3//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 1, 1),                                   # ( 1, 28, 28)
                        nn.ReLU(),                                             # ( 1, 28, 28)
                        nn.Flatten(),                                          # (784)
                        nn.Linear(784, 10)                                     # (10)
                        ])
            
        elif architecture == 'New':
            self.layers = nn.ParameterList([
                        nn.Conv2d(1, 16, 11, padding=11//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 7, padding= 7//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 5, padding= 5//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 3, padding= 3//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Conv2d(16, 16, 3, padding= 3//2),                   # (16, 28, 28)
                        nn.ReLU(),                                             # (16, 28, 28)
                        nn.Flatten(),                                          # (12544)
                        nn.Linear(12544, 784),                                 # (784)
                        nn.ReLU(),                                             # (784)
                        nn.Linear(784, 10)                                     # (10)
                        ])
            
        else:
            raise Exception(f'Architecture not implemented: {architecture}')


    def forward(self, x, mask, epoch_nr, device):

        x = x*mask

        for layer in self.layers:

            x = layer(x)

        return x
