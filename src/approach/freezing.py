import torch
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the freezing baseline """

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        self.model.freeze_all()

    def _get_optimizer(self):
        if len(self.model.heads) == 1:
            return torch.optim.SGD(self.model.parameters(),
                                   lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        else:
            return torch.optim.SGD(self.model.heads[-1].parameters(),
                                   lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        if t == 0:
            self.model.train()
        else:
            self.model.eval()
            self.model.heads[-1].train()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
