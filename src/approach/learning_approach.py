import time
from argparse import ArgumentParser

import torch
import numpy as np

from datasets.exemplars_dataset import ExemplarsDataset
from loggers.exp_logger import ExperimentLogger


class Learning_Appr:
    """ Basic class for implementing learning approaches """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, logger: ExperimentLogger = None,
                 exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.optimizer = None

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """
        Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    # Returns the optimizer
    def _get_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def pre_train_process(self, t, trn_loader):
        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            self.model.heads[-1].train()
            # Loop epochs
            for e in range(self.warmup_epochs):
                # Train
                warmupclock0 = time.time()
                self.train_epoch(t, trn_loader)
                warmupclock1 = time.time()
                trn_loss, trn_acc, _ = self.eval(t, trn_loader)
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    # Contains the epochs loop
    def train(self, t, trn_loader, val_loader):
        best_loss = np.inf
        best_model = self.model.get_copy()
        lr = self.lr
        patience = self.lr_patience

        self.pre_train_process(t, trn_loader)

        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            train_loss, train_acc, _ = self.eval(t, trn_loader)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")

            # Valid
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            print(' Valid: loss={:.3f}, TAw acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

        self.post_train_process(t, trn_loader)

        return

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        pass

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs[0], targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            #print features
            '''
            if len(outputs[1]) == 2: #disentangle network
                for filt in range(int(outputs[1][1].shape[1]/2)):
                    for batch in range(int(outputs[1][1].shape[0])):
                        self.logger.save_PIL(state_dict=outputs[1][1][batch,filt,:,:],task=t, prefix='features_color'+'_batch'+str(batch)+'_filt'+str(filt))
                for filt in range(int(outputs[1][1].shape[1]/2)+1,int(outputs[1][1].shape[1])):
                    for batch in range(int(outputs[1][1].shape[0])):
                        self.logger.save_PIL(state_dict=outputs[1][1][batch,filt,:,:],task=t, prefix='features_shape'+'_batch'+str(batch)+'_filt'+str(filt))
            else: #default resnet (by default output is after avgpool, but we could add a view() with prev. layer for the features)
                model_lastlayer=torch.nn.Sequential(*list(self.model.model.children())[:-2])
                outputs=model_lastlayer(images.to(self.device))
                for filt in range(int(outputs.shape[1])):
                    for batch in range(int(outputs.shape[0])):
                        self.logger.save_PIL(state_dict=outputs[batch,filt,:,:],task=t, prefix='features'+'_batch'+str(batch)+'_filt'+str(filt))
             '''
    # Contains the evaluation code
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs[0], targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs[0], targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
            #print features
            '''
            if len(outputs[1]) == 2: #disentangle network
                for filt in range(int(outputs[1][1].shape[1]/2)):
                    for batch in range(int(outputs[1][1].shape[0])):
                        self.logger.save_PIL(state_dict=outputs[1][1][batch,filt,:,:],task=t, prefix='features_color'+'_batch'+str(batch)+'_filt'+str(filt))
                for filt in range(int(outputs[1][1].shape[1]/2)+1,int(outputs[1][1].shape[1])):
                    for batch in range(int(outputs[1][1].shape[0])):
                        self.logger.save_PIL(state_dict=outputs[1][1][batch,filt,:,:],task=t, prefix='features_shape'+'_batch'+str(batch)+'_filt'+str(filt))
            else: #default resnet (by default output is after avgpool, but we could add a view() with prev. layer for the features)
                model_lastlayer=torch.nn.Sequential(*list(self.model.model.children())[:-2])
                outputs=model_lastlayer(images.to(self.device))
                for filt in range(int(outputs.shape[1])):
                    for batch in range(int(outputs.shape[0])):
                        self.logger.save_PIL(state_dict=outputs[batch,filt,:,:],task=t, prefix='features'+'_batch'+str(batch)+'_filt'+str(filt))
            '''
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # Contains the main Task-Aware and Task-Agnostic metrics
    def calculate_metrics(self, outputs, targets):
        # Task-Aware Multi-Head
        pred = torch.zeros_like(targets.to(self.device))
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    # Returns the loss value
    def criterion(self, t, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
