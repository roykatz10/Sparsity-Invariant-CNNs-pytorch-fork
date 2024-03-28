import torch.nn as nn
import torch



class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.ParameterList([nn.Conv2d(1, 16, 11),                   # (16, 18, 18)
                       nn.ReLU(),                              # (16, 18, 18)
                       nn.Conv2d(16, 16, 7),                   # (16, 12, 12)
                       nn.ReLU(),                              # (16, 12, 12)
                       nn.Conv2d(16, 16, 5),                   # (16,  8,  8)
                       nn.ReLU(),                              # (16,  8,  8)
                       nn.Conv2d(16, 16, 3),                   # (16,  6,  6)
                       nn.ReLU(),                              # (16,  6,  6)
                       nn.Conv2d(16, 16, 3),                   # (16,  4,  4)
                       nn.ReLU(),                              # (16,  4,  4)
                       nn.Flatten(),                           # (256)
                       nn.Linear(256, 128),   # (128)
                       nn.ReLU(),             # (128)
                       nn.Linear(128, 32),    # ( 32)
                       nn.ReLU(),             # ( 32)
                       nn.Linear(32, 10)      # ( 10)
                      ])


    def forward(self, x, mask, epoch_nr, device):

        for layer in self.layers:

            x = layer(x)

        return x
