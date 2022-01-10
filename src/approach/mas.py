import torch
import itertools
from argparse import ArgumentParser
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the Memory Aware Synapses (MAS) approach (global version)
        described in https://arxiv.org/abs/1711.09601
        Original code available at https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger=None, lamb=1,
                 num_samples=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, logger)

        self.lamb = lamb
        self.num_samples = num_samples

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                           if p.requires_grad}

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Eq. 3: lambda is the regularizer trade-off -- In original code: MAS.ipynb block [4]: lambda set to 1
        parser.add_argument('--lamb', default=1, type=float, required=False, help='(default=%(default)s)')
        # Number of samples from train for estimating importance
        parser.add_argument('--num_samples', default=-1, type=int, required=False, help='(default=%(default)s)')
        return parser.parse_known_args(args)

    # Returns the optimizer
    def _get_optimizer(self):
        if len(self.model.heads) > 1:
            return torch.optim.SGD(list(self.model.model.parameters()) + list(self.model.heads[-1].parameters()),
                                   lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        else:
            return torch.optim.SGD(self.model.parameters(),
                                   lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    # Section 4.1: MAS (global) is implemented since the paper shows is more efficient than l-MAS (local)
    def estimate_parameter_importance(self, trn_loader):
        # Initialize importance matrices
        importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters()
                      if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to accumulate L2-loss gradients
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            # MAS allows to any unlabeled data to do the estimation, we choose the current data as in main experiments
            outputs = self.model.forward(images.to(self.device))

            # Page 6: labels not required, "...use the gradients of the squared L2-norm of the learned function output."
            loss = torch.cat(outputs, dim=1).pow(2).sum()

            self.optimizer.zero_grad()
            loss.backward()
            # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(targets)
        # Eq. 2: divide by N total number of samples
        n_samples = n_samples_batches * trn_loader.batch_size
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_importance = self.estimate_parameter_importance(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.importance.keys():
            # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
            self.importance[n] = (self.importance[n] + curr_importance[n])

    # Returns the loss value
    def criterion(self, t, outputs, targets):
        loss_reg = 0
        if t > 0:
            # Eq. 3: memory aware synapses regularizer penalty
            for n, p in self.model.model.named_parameters():
                if n in self.importance.keys():
                    loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2)) / 2
        # since there are no exemplars, the CE loss is only applied to the current training head
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t]) + self.lamb * loss_reg
