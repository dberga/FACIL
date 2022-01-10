from networks.twobranch import twobranch
import torch
import utils
from copy import deepcopy

class twobranch_sum(twobranch):    
    def __init__(self, pretrained=False, backbone='resnet18', num_out=100, togray="mean", scramble=False,select_kernels='all', remove_batchnorm=None):
        self.backbone = backbone
        self.num_out = num_out
        self.togray = togray
        self.scramble = scramble
        self.remove_batchnorm = remove_batchnorm
        self.select_kernels=select_kernels
        #Module init
        super(twobranch_sum, self).__init__(pretrained=pretrained,backbone=backbone,num_out=num_out,togray=togray,scramble=scramble,remove_batchnorm=remove_batchnorm,select_kernels=select_kernels)
        len_fe=len(list(self.feature_extractor.parameters())[-1])
        self.fc=torch.nn.Linear(in_features=len_fe, out_features=self.num_out, bias=True)
        self.head_var = 'fc'
        
    def forward(self,x):
        x1=deepcopy(x)
        if self.scramble is True:
            x1=utils.batch2scramble(deepcopy(x))
        x2=utils.batch2gray(deepcopy(x),transform_type=self.togray)
        return self.forward_sum(x1,x2)
    