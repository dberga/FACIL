import time
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
        described in https://arxiv.org/abs/1611.07725"""

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.00001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger=None,
                 num_exemplars=2000):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, logger)

        self.num_exemplars = num_exemplars
        self.model_old = None

        self.x_train_exemplars = []
        self.y_train_exemplars = []
        self.exemplar_means = []

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to store up to K=2000 exemplars."
        parser.add_argument('--num_exemplars', default=2000, type=int, required=False, help='(default=%(default)s)')
        return parser.parse_known_args(args)

    # Algorithm 1: iCaRL Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, val_loader):
        print('Computing mean of exemplars')
        self.exemplar_means = []
        for cls in range(sum(self.model.task_cls)):
            features = []
            # Extract feature for each exemplar in P_y
            for exemplar in self.x_train_exemplars[cls]:
                image = val_loader.dataset.transform(Image.fromarray(exemplar))
                feats = self.model(image.unsqueeze(0).to(self.device), return_features=True)[1]
                # normalize
                features.append(feats.squeeze() / feats.squeeze().norm())
            features = torch.stack(features)
            # normalize
            self.exemplar_means.append(features.mean(0) / features.mean(0).norm())

    # Algorithm 2: iCaRL Incremental Train
    def train(self, t, trn_loader, val_loader):
        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2
        self.exemplar_means = []

        # Algorithm 3: iCaRL Update Representation
        # Alg. 3. "form combined training set", add exemplars to train_loader
        if t > 0:
            # if dataset is in memory or files type
            if type(trn_loader.dataset.images) is np.ndarray:
                trn_loader.dataset.images = np.vstack([trn_loader.dataset.images, np.vstack(self.x_train_exemplars)])
                trn_loader.dataset.labels.extend(sum(self.y_train_exemplars, []))
            else:
                print('Options for Base Dataset not implemented yet.')
                exit()

        # Alg. 3. "store network outputs with pre-update parameters", we keep the old model instead and do the
        #  distillation like in LwF. This also allows to do data augmentation easily.
        super().train(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT
        print('Managing exemplars')
        clock0 = time.time()

        # number of classes and exemplars per class
        num_cls = sum(self.model.task_cls)
        num_old_cls = sum(self.model.task_cls[:t])
        num_trn_ex_cls = int(np.ceil(self.num_exemplars / num_cls))

        # reduce current exemplar set on old classes
        if t > 0:
            self.reduce_exemplar_set(self.x_train_exemplars, self.y_train_exemplars, num_trn_ex_cls)

        # create current exemplar set on new classes
        extracted_features = self.feature_extraction(trn_loader, val_loader)
        for curr_cls in range(num_old_cls, num_cls):
            # get all indices from current class
            cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # construct exemplar set for current class
            selected, selected_feat = self.construct_exemplar_set(cls_feats, num_trn_ex_cls)
            # add the exemplars to the set
            self.x_train_exemplars.append(trn_loader.dataset.images[cls_ind[selected]])
            self.y_train_exemplars.append([trn_loader.dataset.labels[cls_ind[idx]] for idx in selected])
        # log
        clock1 = time.time()
        print(' | Selected {:d} train exemplars, time={:5.1f}s'.format(sum([len(elem) for elem in self.y_train_exemplars]), clock1 - clock0))

        # store mean of exemplars for each class
        self.compute_mean_of_exemplars(val_loader)

    # Algorithm 4: iCaRL ConstructExemplarSet
    def construct_exemplar_set(self, cls_feats, num_ex_cls):
        # current class mean
        cls_mu = cls_feats.mean(0)
        # select the exemplars closer to the mean of each class
        selected = []
        selected_feat = []
        for k in range(num_ex_cls):
            dist_min = np.inf
            # choose the closest to the mean of the current class
            for item in range(cls_feats.shape[0]):
                if item not in selected:
                    dist = torch.norm(cls_mu - (cls_feats[item] - sum(selected_feat)) / (k + 1.0))
                    if dist < dist_min:
                        dist_min = dist
                        newone = item
                        newonefeat = cls_feats[item]
            selected_feat.append(newonefeat)
            selected.append(newone)
        return selected, selected_feat

    # Algorithm 5: iCaRL ReduceExemplarSet
    def reduce_exemplar_set(self, x_exemplars, y_exemplars, num_ex_cls):
        for cls in range(len(y_exemplars)):
            x_exemplars[cls] = x_exemplars[cls][:num_ex_cls]
            y_exemplars[cls] = y_exemplars[cls][:num_ex_cls]

    # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors, e.g. averages
    #         are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
    def feature_extraction(self, trn_loader, val_loader):
        # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep the same order
        icarl_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False,
                                  num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
        # change transforms to evaluation
        icarl_loader.dataset.transform = val_loader.dataset.transform
        # extract features from the model for all train samples
        extracted_features = []
        with torch.no_grad():
            self.model.eval()
            for images, targets in icarl_loader:
                feats = self.model(images.to(self.device), return_features=True)[1]
                # normalize
                extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
        return (torch.cat(extracted_features)).cpu()

    # --- SIMILAR DISTILLATION APPROACH AS IN LwF --> see approaches/lwf.py --- modifications for iCaRL
    def post_train_process(self, t, trn_loader):
        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    # from LwF: Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        for images, targets in trn_loader:
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
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
                images = images.to(self.device)
                # Forward old model
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images)
                # Forward current model
                outputs, feats = self.model(images, return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # during training, the usual accuracy is computed on the outputs
                if not self.exemplar_means:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                else:
                    hits_taw, hits_tag = self.classify(t, feats, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # classification and distillation terms from Alg. 3. -- original formulation has no trade-off parameter
    def criterion(self, t, outputs, targets, outputs_old):
        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distilation loss for old classes
        if t > 0:
            # The original code does not match with the paper equation, but maybe sigmoid could be removed from g
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            loss += sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in range(sum(self.model.task_cls[:t])))
        return loss
