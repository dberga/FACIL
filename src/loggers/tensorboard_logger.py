from torch.utils.tensorboard import SummaryWriter

from loggers.exp_logger import ExperimentLogger


class Logger(ExperimentLogger):
    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)
        self.tbwriter = SummaryWriter(self.exp_path)

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        self.tbwriter.add_scalar(tag="t{}/{}_{}".format(task, group, name), scalar_value=value, global_step=iter)
        self.tbwriter.file_writer.flush()