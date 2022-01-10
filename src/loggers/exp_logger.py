import importlib
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, log_path, exp_name, begin_time=None):
        self.log_path = log_path
        self.exp_name = exp_name
        self.exp_path = os.path.join(log_path, exp_name)
        if begin_time is None:
            self.begin_time = datetime.now()
        else:
            self.begin_time = begin_time

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        pass

    def log_args(self, args):
        pass

    def log_result(self, array, name):
        pass
   
    def save_model(self, state_dict, task, prefix):
        pass

    def save_model(self, state_dict, task):
        pass


class MultiLogger(ExperimentLogger):
    def __init__(self, log_path, exp_name, loggers=None, save_models=True):
        super(MultiLogger, self).__init__(log_path, exp_name)
        if os.path.exists(self.exp_path):
            print("WARNING: {} already exists!".format(self.exp_path))
        else:
            os.makedirs(os.path.join(self.exp_path, 'models'))
            os.makedirs(os.path.join(self.exp_path, 'results'))
            os.makedirs(os.path.join(self.exp_path, 'features'))

        self.save_models = save_models
        self.loggers = []
        for l in loggers:
            lclass = getattr(importlib.import_module(name='loggers.' + l + '_logger'), 'Logger')
            self.loggers.append(lclass(self.log_path, self.exp_name))

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        if curtime is None:
            curtime = datetime.now()
        for l in self.loggers:
            l.log_scalar(task, iter, name, value, group, curtime)

    def log_args(self, args):
        for l in self.loggers:
            l.log_args(args)

    def log_result(self, array, name):
        for l in self.loggers:
            l.log_result(array, name)
    
    def save_PIL(self, state_dict, task, prefix="features"):
        for l in self.loggers:
            l.save_PIL(state_dict, task, prefix)

    def save_model(self, state_dict, task):
        if self.save_models:
            for l in self.loggers:
                l.save_model(state_dict, task)
