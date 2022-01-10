import time
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from datasets.memory_dataset import MemoryDataset

from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """
    Class implementing the End-to-end Incremental Learning (EEIL) approach
    described in:
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf 

    Ref. code repository:
    https://github.com/fmcp/EndToEndIncrementalLearning

    Helpful code repo:
    https://github.com/arthurdouillard/incremental_learning.pytorch

    """

    def __init__(self, model, device, nepochs=90, lr=0.1, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger=None, num_exemplars=2000, T=2,
                 lr_finetuning=0.01, nepochs_finetuning=40, no_noise_grad=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, logger)

        self.num_exemplars = num_exemplars
        self.model_old = None
        self.T = T
        self.lr_finetuning = lr_finetuning
        self.nepochs_finetuning = nepochs_finetuning
        self.lr_unbalanced = lr
        self.nepochs_unbalanced = nepochs
        self.no_noise_grad = no_noise_grad

        self.x_train_exemplars = []
        self.y_train_exemplars = []

        self._train_epoch = 0
        self._finetune_balanced = None

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # "We store K = 2000 distillation samples in the representative memory
        # for CIFAR-100 and K = 20000 for ImageNet" (page 8)
        parser.add_argument('--num_exemplars', default=2000,
                            type=int, required=False, help='(default=%(default)s)')
        # "Based on our empirical results, we set T to 2 for all our experiments" (page 6)
        parser.add_argument('--T', default=2.0, type=float,
                            required=False, help='(default=%(default)s)')
        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr_finetuning', default=0.01, type=float,
                            required=False, help='(default=%(default)s)')
        parser.add_argument('--nepochs_finetuning', default=40, type=int,
                            required=False, help='(default=%(default)s)')
        parser.add_argument('--no_noise_grad', action='store_true')
        return parser.parse_known_args(args)

    def train(self, t, trn_loader, val_loader):
        if t == 0:  # First task is simple training
            super().train(t, trn_loader, val_loader)
            self._construct_examplar_set(
                trn_loader, self._exemplars_per_class_num())
            return

        # Below procedure is described in paper, page 4:
        # "4. Incremental Learning"
        # Only modification is that instead of preparing examplars
        # before the training, we doing it online using old model.

        # Training process (new + old) - unbalanced training
        self._train_unbalanced(t, trn_loader, val_loader)

        # Balanced fine-tunning (new + old)
        self._train_balanced(t, trn_loader, val_loader)

        # After task training
        # update exemplars
        self._reduce_exemplar_set()
        self._construct_examplar_set(trn_loader)

    def post_train_process(self, t, trn_loader):
        # save model
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def _train_unbalanced(self, t, trn_loader, val_loader):
        self._finetune_balanced = False
        self._train_epoch = 0
        self.lr = self.lr_unbalanced
        self.nepochs = self.nepochs_unbalanced
        loader = self._get_train_loader(trn_loader, False)
        super().train(t, loader, val_loader)

    def _train_balanced(self, t, trn_loader, val_loader):
        self._finetune_balanced = True
        self._train_epoch = 0
        self.lr = self.lr_finetuning
        self.nepochs = self.nepochs_finetuning
        loader = self._get_train_loader(trn_loader, True)
        super().train(t, loader, val_loader)

    def _get_train_loader(self, trn_loader, balanced=False):
        exemplars_ds = MemoryDataset({
            'x': [e for cls_exemplars in self.x_train_exemplars for e in cls_exemplars],
            'y': np.hstack(self.y_train_exemplars)},
            transform=trn_loader.dataset.transform,
            offset=trn_loader.dataset.offset,
            class_indices=trn_loader.dataset.class_indices
        )
        trn_dataset = trn_loader.dataset
        if balanced:
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset,
                                                  indices[:len(exemplars_ds)])

        ds = exemplars_ds + trn_dataset
        return DataLoader(ds,
                          batch_size=trn_loader.batch_size,
                          shuffle=True,
                          num_workers=trn_loader.num_workers,
                          pin_memory=trn_loader.pin_memory)

    def train_epoch(self, t, trn_loader):
        self.model.train()
        for images, targets in trn_loader:
            images = images.to(self.device)

            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images)

            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # "We apply L2-regularization and random noise [21] (with parameters η = 0.3, γ = 0.55)
            # on the gradients to minimize overfitting" (page 8)
            # https://github.com/fmcp/EndToEndIncrementalLearning/blob/master/cnn_train_dag_exemplars.m#L367
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clipgrad)
            if not self.no_noise_grad:
                self._noise_grad(self.model.parameters(), self._train_epoch)
            self.optimizer.step()
        self._train_epoch += 1

    # Returns the loss value
    def criterion(self, t, outputs, targets, outputs_old=None):
        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(
            torch.cat(outputs, dim=1), targets)
        # Distilation loss
        if t > 0 and outputs_old:
            # take into account current head when balanced fine tunning
            last_head_idx = t if self._finetune_balanced else (t - 1)
            for i in range(last_head_idx):
                loss += F.binary_cross_entropy(
                    F.softmax(outputs[i] / self.T, dim=1),
                    F.softmax(outputs_old[i] / self.T, dim=1)
                )
        return loss

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)

    def _exemplars_per_class_num(self):
        num_cls = sum(self.model.task_cls)
        exemplars_per_class = int(np.ceil(self.num_exemplars / num_cls))
        assert exemplars_per_class > 0, "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. Limit of exemplars: {}".format(num_cls, self.num_exemplars)
        return exemplars_per_class

    def _reduce_exemplar_set(self):
        n = self._exemplars_per_class_num()
        assert len(self.x_train_exemplars) == len(self.y_train_exemplars)
        for cls in range(len(self.y_train_exemplars)):
            self.x_train_exemplars[cls] = self.x_train_exemplars[cls][:n]
            self.y_train_exemplars[cls] = self.y_train_exemplars[cls][:n]

    def _construct_examplar_set(self, trn_loader, exemplars_per_class=None):
        """
        "Selection of new samples. This is based on herding selection,
         which produces a sorted list of samples of one class based on
         the distance to the mean sample of that class." (page 5)
        """
        clock0 = time.time()
        if exemplars_per_class is None:
            exemplars_per_class = self._exemplars_per_class_num()
        assert exemplars_per_class > 0, "Number of exemplars per class should be above zero!"
        # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
        ex_sel_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False,
                                   num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
        ex_sel_loader.dataset.transform = trn_loader.dataset.transform

        # extract outputs from the model for all train samples
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            self.model.eval()
            for images, targets in ex_sel_loader:
                extracted_features.append(self.model(images.to(self.device))[0])
                extracted_targets.extend(targets)
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = np.array(extracted_targets)
        # iterate through all classes
        # self.x_train_exemplars = []
        # self.y_train_exemplars = []
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(
                curr_cls)
            assert (exemplars_per_class <= len(cls_ind)
                    ), "Not enough samples to store"
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # calculate the mean
            cls_mu = cls_feats.mean(0)
            # select the exemplars closer to the mean of each class
            selected = []
            selected_feat = []
            for k in range(exemplars_per_class):
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
            # disable transforms for getting raw images
            _orig_transform = trn_loader.dataset.transform
            trn_loader.dataset.transform = Lambda(lambda x: np.array(x))
            selected_images, selected_targets = zip(
                *(trn_loader.dataset[idx] for idx in selected))
            self.x_train_exemplars.append(selected_images)
            self.y_train_exemplars.append(selected_targets)
            trn_loader.dataset.transform = _orig_transform

        # Log
        clock1 = time.time()
        print(' | Selected {:d} train exemplars, time={:5.1f}s'.format(
            sum([len(elem) for elem in self.y_train_exemplars]), clock1 - clock0))
