import torch
import itertools
from argparse import ArgumentParser
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the Elastic Weight Consolidation (EWC) approach
        described in http://arxiv.org/abs/1612.00796 """

    # TODO: check the Ferenc Huszar answer -- https://arxiv.org/pdf/1712.03847.pdf

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger=None, lamb=5000,
                 sampling_type='max_pred', num_samples=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, logger)

        self.lamb = lamb
        self.sampling_type = sampling_type
        self.num_samples = num_samples

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                       if p.requires_grad}

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Eq. 3: "lambda sets how important the old task is compared to the new one"
        parser.add_argument('--lamb', default=5000, type=float, required=False, help='(default=%(default)s)')
        # TODO: check where this options were mentioned
        parser.add_argument('--sampling_type', default='max_pred', type=str, required=False,
                            choices=['true', 'max_pred', 'multinomial'], help='(default=%(default)s)')
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

    def compute_fisher_matrix_diag(self, trn_loader):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters()
                  if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to compute the fisher information
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):

            outputs = self.model.forward(images.to(self.device))

            if self.sampling_type == 'true':
                # Use the labels to compute the gradients based on the CE-loss with the ground truth
                preds = targets.to(self.device)
            elif self.sampling_type == 'max_pred':
                # Not use labels and compute the gradients related to the prediction the model has learned
                preds = torch.cat(outputs, dim=1).argmax(1).flatten()
            elif self.sampling_type == 'multinomial':
                # Use a multinomial sampling to compute the gradients
                probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), preds)
            self.optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)
        # Apply mean across all samples
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.fisher.keys():
            self.fisher[n] = (self.fisher[n] + curr_fisher[n])

    # Returns the loss value
    def criterion(self, t, outputs, targets):
        loss_reg = 0
        if t > 0:
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.model.model.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
        # since there are no exemplars, the CE loss is only applied to the current training head
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t]) + self.lamb*loss_reg
