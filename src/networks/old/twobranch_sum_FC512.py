from networks.twobranch_sum import twobranch_sum
import torch

class twobranch_sum_FC512(twobranch_sum):
    def __init__(self, pretrained=False, backbone='resnet18',num_out=100, togray="mean",scramble=False,select_kernels='all', remove_batchnorm=None):
        self.backbone = backbone
        self.num_out = num_out
        self.togray = togray
        self.scramble = scramble
        self.remove_batchnorm = remove_batchnorm
        self.select_kernels=select_kernels
        #twobranch init
        super(twobranch_sum_FC512, self).__init__(pretrained=pretrained,backbone=backbone,num_out=num_out,togray=togray,scramble=scramble,remove_batchnorm=remove_batchnorm,select_kernels=select_kernels)
        #change each branch heads
        self.feature_extractor._modules['fc']=torch.nn.Linear(in_features=512, out_features=512, bias=True)
        self.feature_extractor2._modules['fc']=torch.nn.Linear(in_features=512, out_features=512, bias=True)
        #HEAD (this will be erased in case we do not put --not_remove_existing_head)
        self.fc=torch.nn.Linear(in_features=512, out_features=num_out, bias=True) 
    '''
    def forward(self,x1,x2):
        return self.forward_concat(x1,x2)
    '''
    