from networks.twobranch import twobranch
import torch
import utils
from copy import deepcopy

class twobranch_concat(twobranch):
    def forward(self,x):
        x1=deepcopy(x)
        if self.scramble is True:
            x1=utils.batch2scramble(deepcopy(x))
        x2=utils.batch2gray(deepcopy(x),transform_type=self.togray)
        return self.forward_concat(x1,x2)