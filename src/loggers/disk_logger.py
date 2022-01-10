import datetime
import json
import os
import sys

import numpy as np
import torch
import torchvision

from loggers.exp_logger import ExperimentLogger


class Logger(ExperimentLogger):
    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)

        self.begin_time_str = self.begin_time.strftime("%Y-%m-%d-%H-%M")

        # Duplicate standard outputs
        sys.stdout = FileOutputDuplicator(sys.stdout,
                                          os.path.join(self.exp_path, 'stdout-{}.txt'.format(self.begin_time_str)), 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr,
                                          os.path.join(self.exp_path, 'stderr-{}.txt'.format(self.begin_time_str)), 'w')

        # Raw log file
        self.raw_log_file = open(os.path.join(self.exp_path, "raw_log.txt"), 'a')

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        if curtime is None:
            curtime = datetime.now()

        # Raw dump
        entry = {"task": task, "iter": iter, "name": name, "value": value, "group": group,
                 "time": curtime.strftime("%Y-%m-%d-%H-%M")}
        self.raw_log_file.write(json.dumps(entry, sort_keys=True) + ",\n")
        self.raw_log_file.flush()

    def log_args(self, args):
        with open(os.path.join(self.exp_path, 'args-{}.txt'.format(self.begin_time_str)), 'w') as f:
            json.dump(args.__dict__, f, separators=(',\n', ' : '), sort_keys=True)

    def log_result(self, array, name):
        if array.ndim <= 1:
            array = array[None]
        np.savetxt(os.path.join(self.exp_path, 'results', '{}.txt'.format(name)),
                   array, '%.6f')

    def save_PIL(self, state_dict, task, prefix="features"):
        #tensor2image=torchvision.transforms.ToPILImage()
        im=torchvision.transforms.ToPILImage()(state_dict.cpu())
        im.save(os.path.join(self.exp_path,'features',prefix+"_t"+str(task+1)+".png"), "PNG")

    def save_model(self, state_dict, task):
        torch.save(state_dict, os.path.join(self.exp_path, "models", "task{}.ckpt".format(task)))

    def __del__(self):
        self.raw_log_file.close()


class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()
