import torch.nn as nn
import torch

class Train:

    def __init__(self, model, data_loader, optim, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device

    def run_epoch(self, lr_updater, epoch_nr, iteration_loss=False, ):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0

        for step, batch_data in enumerate(self.data_loader):

            # Get the inputs and labels
            # batch_data[0] (128, 1, 28, 28)
            # batch_data[1] (128)
            
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            # mask = (inputs>-100).float().to(self.device)
            mask = (torch.rand(inputs.shape) > 0.7).float().to(self.device)

            outputs = self.model(inputs, mask)

            if step < 3:
                print('labels', labels)
                print('output', torch.argmax(outputs, dim = -1) )
                print('output2', outputs)

            # Loss computation
            # print('output', outputs.shape)
            # print('label', labels.shape)
            # print('mask', mask.shape)
            # raise
            
            # loss = (self.criterion(outputs, labels)*mask.detach()).sum()/mask.sum()

            # print(outputs.requires_grad)
            # print(labels.requires_grad)
            # raise
            # print(outputs.shape)
            # print(nn.functional.softmax(outputs, dim=-1).argmax(dim=-1).shape)
            # print(nn.functional.softmax(outputs, dim=-1))
            # raise
            # print(nn.functional.softmax(outputs, dim=-1).argmax(dim=-1))
            # print(labels)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # loss = self.criterion(outputs.squeeze().float(), labels.float())
            # loss.requires_grad = True
            # print(loss.shape)
            # print(outputs.squeeze().float().shape)
            # print(labels.float().shape)
            # raise
            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            lr_updater.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader)
