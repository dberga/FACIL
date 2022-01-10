import torch
from copy import deepcopy
from argparse import ArgumentParser

from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the Learning Without Forgetting approach
        described in https://arxiv.org/abs/1606.09282 """

    # Weight decay of 0.0005 is used in the original article (page 4).
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger=None, lamb=1, T=2):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, logger)

        self.model_old = None
        self.lamb = lamb
        self.T = T

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Page 5: "λ is a loss balance weight, set to 1 for most our experiments. Making λ larger will favor the old
        #  task performance over the new task’s, so we can obtain a old-task-new-task performance line by changing λ."
        parser.add_argument('--lamb', default=1, type=float, required=False, help='(default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations."
        parser.add_argument('--T', default=2, type=int, required=False, help='(default=%(default)s)')
        return parser.parse_known_args(args)

    # We do not do warm-up of the new layer (paper reports that improves performance, but negligible)
    # Page 4:  "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."


    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        for images, targets in trn_loader:
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    # Contains the evaluation code
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    # Returns the loss value
    def criterion(self, t, outputs, targets, targets_old):
        # Knowledge distillation loss for all previous tasks
        loss_dist = 0
        if t > 0:
            loss_dist += self.cross_entropy(torch.cat(outputs[:t], dim=1), torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T)

        # Current cross-entropy loss with lambda trade-off
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t]) + self.lamb*loss_dist
