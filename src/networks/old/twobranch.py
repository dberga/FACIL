import torch
import importlib
import pdb
import utils
#import numpy as np
from argparse import ArgumentParser
import pdb
from copy import deepcopy

class twobranch(torch.nn.Module):
    
    def __init__(self, pretrained=False, backbone='resnet18', num_out=100, togray="mean",scramble=False,select_kernels='all', remove_batchnorm=None):
        #Module init
        super(twobranch, self).__init__()
        self.backbone = backbone
        self.num_out = num_out
        self.togray = togray
        self.scramble = scramble
        self.remove_batchnorm = remove_batchnorm
        self.select_kernels=select_kernels
        #load backbone
        try:
            tvnet = getattr(importlib.import_module(name='torchvision.models'), self.backbone)
            init_model=tvnet(pretrained=False) #load backbone
        except:
            net = getattr(importlib.import_module(name='networks.' + self.backbone), self.backbone)
            init_model = net(pretrained=False)
        
        #BRANCH 1 - RGB
        self.feature_extractor = utils.replace_kernels(deepcopy(init_model),kernel_size=(1,1),select=self.select_kernels) #replace conv2d kernels by 1x1
        if self.remove_batchnorm == True:
            self.feature_extractor = utils.replace_batchnorm(deepcopy(self.feature_extractor),select='all') #remove all batchnorm
        #BRANCH 2 - GRAY
        self.feature_extractor2 = init_model #load backbone
        if self.togray is not None and "keepdim" in str(self.togray):
            self.feature_extractor2 = utils.replace_dim(self.feature_extractor2,target_dim=1,select='first')
        
        #HEAD (this will be erased in case we do not put --not_remove_existing_head)
        len_fe=len(list(self.feature_extractor.parameters())[-1])
        len_fe2=len(list(self.feature_extractor2.parameters())[-1])
        self.fc=torch.nn.Linear(in_features=len_fe+len_fe2, out_features=self.num_out, bias=True)
        self.head_var = 'fc'
        
        
    def forward_concat(self,x1,x2):
        features = self.feature_extractor(x1)
        features = features.view(features.size(0),-1) #flatten
        features2 = self.feature_extractor2(x2)
        features2 = features2.view(features2.size(0),-1) #flatten
        grouped = torch.cat((features,features2),dim=1)
        output = grouped.view(grouped.size(0),-1) #flatten
        output = self.fc(grouped)
        return output
    
    def forward_sum(self,x1,x2):
        features = self.feature_extractor(x1)
        features = features.view(features.size(0),-1) #flatten
        features2 = self.feature_extractor2(x2)
        features2 = features2.view(features2.size(0),-1) #flatten
        grouped = features.add(features2)
        output = grouped.view(grouped.size(0),-1) #flatten
        output = self.fc(grouped)
        return output
    '''
    def forward(self,x1,x2):
        return self.forward_concat(x1,x2)
    '''
    def forward(self,x):
        x1=deepcopy(x)
        if self.scramble is True:
            x1=utils.batch2scramble(deepcopy(x))
        x2=utils.batch2gray(deepcopy(x),transform_type=self.togray)
        return self.forward_concat(x1,x2)

    
    @staticmethod
    def extra_parser(args):
        args_new=['--backbone','--num_out', '--togray','--scramble','--select_kernels','--remove_batchnorm']
        args=[arg for arg in args if arg.split(sep='=')[0] in args_new]
        parser = ArgumentParser()
        parser.add_argument('--backbone', default='resnet18', type=str, required=False, help='(default=%(default)s)')
        parser.add_argument('--num_out', default=100, type=int, required=False, help='(default=%(default)s)')
        parser.add_argument('--togray', default=None, type=str, help='(default=%(default)s)')
        parser.add_argument('--scramble', default=False, type=bool, help='(default=%(default)s)')
        parser.add_argument('--select_kernels', default='all', type=str, help='(default=%(default)s)')
        parser.add_argument('--remove_batchnorm', default=False, type=bool, help='(default=%(default)s)')
        return parser.parse_args(args)
    
