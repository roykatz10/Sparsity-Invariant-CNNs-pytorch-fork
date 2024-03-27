import torch.nn as nn
from matplotlib import pyplot as plt

class Train:

    def __init__(self, model, data_loader, optim, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device

    def run_epoch(self, lr_updater, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for step, batch_data in enumerate(self.data_loader):

            # Get the inputs and labels
            # batch_data[0] (128, 1, 28, 28)
            # batch_data[1] (128)
            
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation

            # plt.figure()
            # plt.imshow(inputs[0].squeeze().cpu().detach().numpy())
            # plt.colorbar()
            # plt.title('x')

            # plt.figure()
            # plt.imshow((inputs>0)[0].squeeze().cpu().detach().numpy())
            # plt.colorbar()
            # plt.title('mask > 0')

            # plt.figure()
            # plt.imshow((inputs>=0)[0].squeeze().cpu().detach().numpy())
            # plt.colorbar()
            # plt.title('mask >= 0')
            # plt.show()

            # raise
            

            mask = (inputs>=0).float()
            outputs = self.model(inputs, mask)

            # Loss computation
            
            # loss = (self.criterion(outputs, labels)*mask.detach()).sum()/mask.sum()
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            lr_updater.step()

            # Keep track of loss for current epoch
            acc = (outputs.max(dim=-1)[1] == labels).sum().item() / len(labels)

            epoch_loss += loss.item()
            epoch_acc += acc

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), epoch_acc / len(self.data_loader)
