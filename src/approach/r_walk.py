import time
import torch
import random
import itertools
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the Riemannian Walk approach described in
        http://openaccess.thecvf.com/content_ECCV_2018/papers/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.pdf """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger=None, lamb=1, alpha=0.5,
                 damping=0.1, fim_sampling_type='max_pred', fim_num_samples=-1, num_exemplars=200,
                 exemplar_selection='herding'):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, logger)

        self.lamb = lamb
        self.alpha = alpha
        self.damping = damping
        self.sampling_type = fim_sampling_type
        self.num_samples = fim_num_samples
        self.num_exemplars = num_exemplars
        self.exemplar_selection = exemplar_selection

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Page 7: "task-specific parameter importance over the entire training trajectory."
        self.w = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store scores and fisher information
        self.scores = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters() if p.requires_grad}

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--alpha', default=0.5, type=float, required=False, help='(default=%(default)s)') # in [0,1]
        parser.add_argument('--damping', default=0.1, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--fim_num_samples', default=-1, type=int, required=False, help='(default=%(default)s)')
        parser.add_argument('--fim_sampling_type', default='max_pred', type=str, required=False,
                            choices=['true', 'max_pred', 'multinomial'], help='(default=%(default)s)')
        parser.add_argument('--num_exemplars', default=200, type=int, required=False, help='(default=%(default)s)')
        # TODO: implemented random uniform and herding, they also propose two more sampling strategies
        parser.add_argument('--exemplar_selection', default='random', type=str, choices=['herding', 'random'],
                            required=False, help='(default=%(default)s)')
        return parser.parse_known_args(args)

    # Returns the optimizer
    def _get_optimizer(self):
        if len(self.model.heads) > 1:
            return torch.optim.SGD(list(self.model.model.parameters()) + list(self.model.heads[-1].parameters()),
                                   lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        else:
            return torch.optim.SGD(self.model.parameters(),
                                   lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, t, trn_loader, val_loader):
        # number of classes and buffer samples per class
        num_cls = sum(self.model.task_cls)
        num_trn_ex_cls = int(np.ceil(self.num_exemplars / num_cls))

        # add exemplars to train_loader
        if self.num_exemplars > 0 and t > 0:
            # if dataset is in memory or files type
            if type(trn_loader.dataset.images) is np.ndarray:
                trn_loader.dataset.images = np.vstack([trn_loader.dataset.images, np.vstack(self.x_train_exemplars)])
                trn_loader.dataset.labels.extend(sum(self.y_train_exemplars, []))
            else:
                print('Adding exemplars in Base Dataset is not implemented yet.')
                exit()
        # RESUME DEFAULT TRAINING -- contains the epochs loop
        super().train(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        if self.num_exemplars > 0:
            print('Select training exemplars')
            clock0 = time.time()
            if self.exemplar_selection == 'random':
                # iterate through all existing classes
                self.x_train_exemplars = []
                self.y_train_exemplars = []
                for curr_cls in range(num_cls):
                    # get all indices from current class -- check if there are exemplars from previous task in loader
                    cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
                    assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
                    assert (num_trn_ex_cls <= len(cls_ind)), "Not enough samples to store"
                    # select the exemplars randomly
                    selected = random.sample(list(cls_ind), num_trn_ex_cls)
                    # add the exemplars to the buffer
                    self.x_train_exemplars.append(trn_loader.dataset.images[selected])
                    self.y_train_exemplars.append([trn_loader.dataset.labels[idx] for idx in selected])
            elif self.exemplar_selection == 'herding':
                # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
                ex_sel_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False,
                                           num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
                ex_sel_loader.dataset.transform = val_loader.dataset.transform

                # extract outputs from the model for all train samples
                extracted_features = []
                with torch.no_grad():
                    self.model.eval()
                    for images, targets in ex_sel_loader:
                        extracted_features.append(self.model(images.to(self.device))[0])
                extracted_features = (torch.cat(extracted_features)).cpu()

                # iterate through all existing classes
                self.x_train_exemplars = []
                self.y_train_exemplars = []
                for curr_cls in range(num_cls):
                    # get all indices from current class -- check if there are exemplars from previous task in loader
                    cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
                    assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
                    assert (num_trn_ex_cls <= len(cls_ind)), "Not enough samples to store"
                    # get all extracted features for current class
                    cls_feats = extracted_features[cls_ind]
                    # calculate the mean
                    cls_mu = cls_feats.mean(0)
                    # select the exemplars closer to the mean of each class
                    selected = []
                    selected_feat = []
                    for k in range(num_trn_ex_cls):
                        # fix this to the dimension of the model features
                        sum_others = torch.zeros(cls_feats.shape[1])
                        for j in selected_feat:
                            sum_others += j / (k + 1)
                        dist_min = np.inf
                        # choose the closest to the mean of the current class
                        for item in cls_ind:
                            if item not in selected:
                                feat = extracted_features[item]
                                dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                                if dist < dist_min:
                                    dist_min = dist
                                    newone = item
                                    newonefeat = feat
                        selected_feat.append(newonefeat)
                        selected.append(newone)
                    # add the exemplars to the buffer
                    self.x_train_exemplars.append(trn_loader.dataset.images[selected])
                    self.y_train_exemplars.append([trn_loader.dataset.labels[idx] for idx in selected])
            # Log
            clock1 = time.time()
            print(' | Selected {:d} train exemplars, time={:5.1f}s'.format(
                                                  sum([len(elem) for elem in self.y_train_exemplars]), clock1 - clock0))

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        for images, targets in trn_loader:
            # store current model
            curr_feat_ext = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

            # Forward current model
            outputs = self.model(images.to(self.device))

            # cross-entropy loss on current task
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets.to(self.device))
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # store gradients without regularization term
            unreg_grads = {n: p.grad.clone().detach() for n, p in self.model.model.named_parameters()
                            if p.grad is not None}
            # apply loss with path integral regularization
            loss = self.criterion(t, outputs, targets.to(self.device))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Page 7: "accumulate task-specific parameter importance over the entire training trajectory"
            #  "the parameter importance is defined as the ratio of the change in the loss function to the distance
            #  between the conditional likelihod distributions per step in the parameter space."
            with torch.no_grad():
                for n, p in self.model.model.named_parameters():
                    if n in unreg_grads.keys():
                        self.w[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

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
            # Page 6: "the Fisher component [...] is the expected square of the loss gradient w.r.t the i-th parameter."
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

        # calculate Fisher Information Matrix
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)

        # Eq. 10: efficiently update Fisher Information Matrix
        for n in self.fisher.keys():
            self.fisher[n] = self.alpha * curr_fisher[n] + (1 - self.alpha) * self.fisher[n]

        # Page 7: Optimization Path-based Parameter Importance: importance scores computation
        curr_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters()
                      if p.requires_grad}
        with torch.no_grad():
            curr_params = {n: p for n, p in self.model.model.named_parameters() if p.requires_grad}
            for n, p in self.scores.items():
                curr_score[n] = self.w[n] / (self.fisher[n] * ((curr_params[n] - self.older_params[n]) ** 2) + self.damping)
                self.w[n].zero_()
                # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                curr_score[n] = torch.nn.functional.relu(curr_score[n])
        # Page 8: alleviating regularization getting increasingly rigid by averaging scores
        for n, p in self.scores.items():
            self.scores[n] = (self.scores[n] + curr_score[n]) / 2

    # Returns the loss value
    def criterion(self, t, outputs, targets):
        loss_reg = 0
        if t > 0:
            # Eq. 9: final objective function
            for n, p in self.model.model.named_parameters():
                loss_reg += torch.sum((self.fisher[n] + self.scores[n]) * (p - self.older_params[n]).pow(2))
        # since there are no exemplars, the CE loss is only applied to the current training head
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets) + self.lamb * loss_reg
