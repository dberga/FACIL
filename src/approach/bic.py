from copy import deepcopy
from argparse import ArgumentParser

import time
import torch
import random
import numpy as np
from datasets import memory_dataset as memd
from torch.utils.data import DataLoader
from .learning_approach import Learning_Appr

class Appr(Learning_Appr):
    """ Class implementing the Bias Correction (BiC) approach described in Large Scale Incremental Learning
        http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf"""

    # Inspiration from https://github.com/wuyuebupt/LargeScaleIncrementalLearning/

    # Sec. 6.2. weight decay for CIFAR-100 is 0.0002, for ImageNet-1000 and Celeb-10000 is 0.0001
    # In their code is specified that momentum is always 0.9
    def __init__(self, model, device, nepochs=250, lr=0.1, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0002, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger=None,
                 num_exemplars=2000, val_percentage=0.1, bias_epochs=2, exemplar_selection='herding', T=2):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, logger)

        self.num_exemplars = num_exemplars
        self.val_percentage = val_percentage
        self.bias_epochs = bias_epochs
        self.exemplar_selection = exemplar_selection
        self.model_old = None
        self.T = T
        self.bias_layers = []

        self.x_train_exemplars = []
        self.y_train_exemplars = []
        self.x_valid_exemplars = []
        self.y_valid_exemplars = []

    # Returns a parser containing the approach specific parameters
    # Sec. 3. "lambda is set to n / (n+m)" where n=num_old_classes and m=num_new_classes - so lambda is not a param
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Sec. 6.1. CIFAR-100: 2,000 exemplars, ImageNet-1000: 20,000 exemplars, Celeb-10000: 50,000 exemplars
        parser.add_argument('--num_exemplars', default=2000, type=int, required=False, help='(default=%(default)s)')
        # Sec. 6.1. "the ratio of train/validation split on the exemplars is 9:1 for CIFAR-100 and ImageNet-1000"
        parser.add_argument('--val_percentage', default=0.1, type=float, required=False, help='(default=%(default)s)')
        # In their code they define epochs_per_eval=100 and epoch_val_times=2, making a total of 200 bias epochs
        parser.add_argument('--bias_epochs', default=200, type=int, required=False, help='(default=%(default)s)')
        parser.add_argument('--exemplar_selection', default='herding', type=str, choices=['herding', 'random'],
                            required=False, help='(default=%(default)s)')
        # Sec. 6.2. "The temperature scalar T in Eq. 1 is set to 2 by following [13,2]."
        parser.add_argument('--T', default=2, type=int, required=False, help='(default=%(default)s)')
        return parser.parse_known_args(args)

    # Some parts could go into self.pre_train_process() or self.post_train_process(),
    #  but we leave it here for readability, and because we need to access val_loader
    def train(self, t, trn_loader, val_loader):
        # number of classes and proto samples per class
        num_cls = sum(self.model.task_cls)
        num_old_cls = sum(self.model.task_cls[:t])
        num_trn_ex_cls = int(np.ceil((1 - self.val_percentage) * self.num_exemplars / num_cls))
        num_val_ex_cls = int(np.ceil(self.val_percentage * self.num_exemplars / num_cls))
        # add a bias layer for the new classes
        self.bias_layers.append(BiasLayer().to(self.device))

        # STAGE 0: EXEMPLAR MANAGEMENT -- select subset of validation to use in Stage 2 -- val_old, val_new (Fig.2)
        print('Stage 0: Select exemplars from validation')
        clock0 = time.time()

        # Remove extra exemplars from previous classes -- val_old
        if t > 0:
            num_old_ex_cls = int(np.ceil(self.val_percentage * self.num_exemplars / num_old_cls))
            for cls in range(num_old_cls):
                assert (len(self.y_valid_exemplars[cls]) == num_old_ex_cls)
                self.x_valid_exemplars[cls] = self.x_valid_exemplars[cls][:num_val_ex_cls]
                self.y_valid_exemplars[cls] = self.y_valid_exemplars[cls][:num_val_ex_cls]

        # Add new exemplars for current classes -- val_new
        non_selected = []
        for curr_cls in range(num_old_cls, num_cls):
            self.x_valid_exemplars.append([])
            self.y_valid_exemplars.append([])
            # get all indices from current class
            cls_ind = np.where(np.asarray(val_loader.dataset.labels) == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (num_val_ex_cls <= len(cls_ind)), "Not enough samples to store for class {:d}".format(curr_cls)
            # add samples to the exemplar list
            self.x_valid_exemplars[curr_cls] = val_loader.dataset.images[cls_ind[:num_val_ex_cls]]
            self.y_valid_exemplars[curr_cls] = [val_loader.dataset.labels[idx] for idx in cls_ind[:num_val_ex_cls]]
            non_selected.extend(cls_ind[num_val_ex_cls:])
        # remove selected samples from the validation data used during training
        val_loader.dataset.images = val_loader.dataset.images[non_selected]
        val_loader.dataset.labels = [val_loader.dataset.labels[idx] for idx in non_selected]
        clock1 = time.time()
        print(' > Selected {:d} validation exemplars, time={:5.1f}s'.format(sum([len(elem) for elem in self.y_valid_exemplars]), clock1 - clock0))

        # add exemplars to train_loader -- train_new + train_old (Fig.2)
        if t > 0:
            trn_loader.dataset.images = np.vstack([trn_loader.dataset.images, np.vstack(self.x_train_exemplars)])
            trn_loader.dataset.labels.extend(sum(self.y_train_exemplars, []))

        # STAGE 1: DISTILLATION
        print('Stage 1: Training model with distillation')
        super().train(t, trn_loader, val_loader)
        # From LwF: Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        # STAGE 2: BIAS CORRECTION
        if t > 0:
            print('Stage 2: Training bias correction layers')
            # Fill bic_val_loader with validation protoset
            data = {'x': np.vstack(self.x_valid_exemplars), 'y': sum(self.y_valid_exemplars, [])}
            bic_val_dataset = memd.MemoryDataset(data, trn_loader.dataset.transform)
            bic_val_loader = DataLoader(bic_val_dataset, batch_size=trn_loader.batch_size, shuffle=True,
                                        num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

            # bias optimization on validation
            self.model.eval()
            # Allow to learn the alpha and beta for the current task
            self.bias_layers[t].alpha.requires_grad = True
            self.bias_layers[t].beta.requires_grad = True

            # Their code does not mention this optimizer parameters, we use the ones from the code
            #  line 654 from https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/master/resnet.py
            bic_optimizer = torch.optim.SGD(self.bias_layers[t].parameters(), lr=self.lr, momentum=0.9)
            # Loop epochs
            for e in range(self.bias_epochs):
                # Train bias correction layers
                clock0 = time.time()
                total_loss, total_acc = 0, 0
                for inputs, targets in bic_val_loader:
                    # Forward current model
                    with torch.no_grad():
                        outputs = self.model(inputs.to(self.device))
                        old_cls_outs = self.bias_forward(outputs[:t])
                    new_cls_outs = self.bias_layers[t](outputs[t])
                    pred_all_classes = torch.cat([torch.cat(old_cls_outs, dim=1), new_cls_outs], dim=1)
                    # Eqs. 4-5: outputs from previous tasks are not modified (any alpha or beta from those is fixed),
                    #           only alpha and beta from the new task is learned. No temperature scaling used.
                    loss = torch.nn.functional.cross_entropy(pred_all_classes, targets.to(self.device))
                    # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
                    loss += 0.1 * ((self.bias_layers[t].beta[0] ** 2) / 2)
                    # Log
                    total_loss += loss.item() * len(targets)
                    total_acc += ((pred_all_classes.argmax(1) == targets.to(self.device)).float()).sum().item()
                    # Backward
                    bic_optimizer.zero_grad()
                    loss.backward()
                    bic_optimizer.step()
                clock1 = time.time()
                # reducing the amount of verbose
                if (e + 1) % (self.bias_epochs / 4) == 0:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: loss={:.3f}, TAg acc={:5.1f}% |'.format(
                          e + 1, clock1 - clock0, total_loss / len(bic_val_loader.dataset.labels),
                          100 * total_acc / len(bic_val_loader.dataset.labels)))
            # Fix alpha and beta after learning them
            self.bias_layers[t].alpha.requires_grad = False
            self.bias_layers[t].beta.requires_grad = False

        # Print all alpha and beta values
        for task in range(t + 1):
            print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,
                  self.bias_layers[task].alpha.item(), self.bias_layers[task].beta.item()))

        # STAGE 3: EXEMPLAR MANAGEMENT
        print('Stage 3: Select exemplars from training')
        clock0 = time.time()
        self.x_train_exemplars = []
        self.y_train_exemplars = []

        if self.exemplar_selection == 'random':
            # iterate through all existing classes
            for curr_cls in range(num_cls):
                # get all indices from current class -- check if there are exemplars from previous task in the loader
                cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
                assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
                assert (num_trn_ex_cls <= len(cls_ind)), "Not enough samples to store"
                # select the exemplars randomly
                selected = random.sample(list(cls_ind), num_trn_ex_cls)
                # add the exemplars to the buffer
                self.x_train_exemplars.append(trn_loader.dataset.images[selected])
                self.y_train_exemplars.append([trn_loader.dataset.labels[idx] for idx in selected])

        elif self.exemplar_selection == 'herding':
            # copy the dataset so it can be fixed to go sequentially (shuffle=False), this allows to keep the same order
            bic_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False,
                                    num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            bic_loader.dataset.transform = val_loader.dataset.transform
            # extract outputs from the model for all train samples
            extracted_features = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in bic_loader:
                    extracted_features.append(self.model(images.to(self.device))[0])
            extracted_features = (torch.cat(extracted_features)).cpu()
            # iterate through all existing classes
            for curr_cls in range(num_cls):
                # get all indices from current class -- check if there are exemplars from previous task in the trn_loader
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
                # add the exemplars to the protoset
                self.x_train_exemplars.append(trn_loader.dataset.images[selected])
                self.y_train_exemplars.append([trn_loader.dataset.labels[idx] for idx in selected])
        # Log
        clock1 = time.time()
        print(' > Selected {:d} train exemplars, time={:5.1f}s'.format(sum([len(elem) for elem in self.y_train_exemplars]), clock1 - clock0))

    # utility function --- inspired by https://github.com/sairin1202/BIC
    def bias_forward(self, outputs):
        bic_outputs = []
        for m in range(len(outputs)):
            bic_outputs.append(self.bias_layers[m](outputs[m]))
        return bic_outputs

    # --- SAME DISTILLATION APPROACH AS IN LwF --> see approaches/lwf.py --- modifications for bias correction (BiC)
    # from LwF: Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        for images, targets in trn_loader:
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device))
                targets_old = self.bias_forward(targets_old)  # apply bias correction
            # Forward current model
            outputs = self.model(images.to(self.device))
            outputs = self.bias_forward(outputs)  # apply bias correction
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    # from LwF: Contains the evaluation code
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                    targets_old = self.bias_forward(targets_old)  # apply bias correction
                # Forward current model
                outputs = self.model(images.to(self.device))
                outputs = self.bias_forward(outputs)  # apply bias correction
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # from LwF: calculates cross-entropy with temperature scaling
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

    # from LwF: Returns the loss value
    def criterion(self, t, outputs, targets, targets_old):
        # Knowledge distillation loss for all previous tasks
        loss_dist = 0
        if t > 0:
            loss_dist += self.cross_entropy(torch.cat(outputs[:t], dim=1), torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T)
        # trade-off - if we use the lambda from the paper, use the line below.
        lamb = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
        return (1.0 - lamb) * torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets) + lamb * loss_dist


class BiasLayer(torch.nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False, device="cuda"))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device="cuda"))

    def forward(self, x):
        return self.alpha * x + self.beta
