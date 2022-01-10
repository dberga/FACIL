from networks.twobranch_concat import twobranch_concat
import torch
import utils
from copy import deepcopy


class twobranch_concat_LF(twobranch_concat):
    #*args, **kwargs
    def __init__(self, pretrained=False, backbone='resnet18',num_out=100, togray="mean",scramble=False,select_kernels='all', remove_batchnorm=None):
        self.backbone = backbone
        self.num_out = num_out
        self.togray = togray
        self.scramble = scramble
        self.remove_batchnorm = remove_batchnorm
        self.select_kernels=select_kernels
        #twobranch init
        super(twobranch_concat, self).__init__(pretrained=pretrained,backbone=backbone,num_out=num_out,togray=togray,scramble=scramble,remove_batchnorm=remove_batchnorm,select_kernels=select_kernels)
        #remove last layers from each branch
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1]) #remove fc
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1]) #remove avgpool
        self.feature_extractor2 = torch.nn.Sequential(*list(self.feature_extractor2.children())[:-1]) #remove fc
        self.feature_extractor2 = torch.nn.Sequential(*list(self.feature_extractor2.children())[:-1]) #remove avgpool
        #add dimreduc conv1x1
        self.feature_extractor=torch.nn.Sequential(self.feature_extractor,torch.nn.Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.feature_extractor2=torch.nn.Sequential(self.feature_extractor2,torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False))
        #HEAD (this will be erased in case we do not put --not_remove_existing_head)
        len_fe=len(list(self.feature_extractor.parameters())[-1])
        len_fe2=len(list(self.feature_extractor2.parameters())[-1])
        self.fc=torch.nn.Linear(in_features=(7*7*160), out_features=25, bias=True) #head
        #self.fc=torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=(7,7),stride=None),torch.nn.Linear(in_features=(160), out_features=25, bias=True))
    '''
    def forward(self,x1,x2):
        return self.forward_concat(x1,x2)
    '''
    
    def forward_concat_local(self,x1,x2):
        features = self.feature_extractor(x1)
        features2 = self.feature_extractor2(x2)
        grouped = torch.cat((features,features2),dim=1)
        output = grouped.view(grouped.size(0),-1) #flatten
        #import pdb; pdb.set_trace()
        output = self.fc(output)
        return output
    
    def forward(self,x):
        x1=deepcopy(x)
        if self.scramble is True:
            x1=utils.batch2scramble(deepcopy(x))
        x2=utils.batch2gray(deepcopy(x),transform_type=self.togray)
        return self.forward_concat_local(x1,x2)
