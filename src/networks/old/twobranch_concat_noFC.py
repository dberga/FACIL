from networks.twobranch_concat import twobranch_concat
import torch


class twobranch_concat_noFC(twobranch_concat):
    #*args, **kwargs
    def __init__(self, pretrained=False, backbone='resnet18',num_out=100, togray="mean",scramble=False,select_kernels='all', remove_batchnorm=None):
        self.backbone = backbone
        self.num_out = num_out
        self.togray = togray
        self.scramble = scramble
        self.remove_batchnorm = remove_batchnorm
        self.select_kernels=select_kernels
        #twobranch init
        super(twobranch_concat_noFC, self).__init__(pretrained=pretrained,backbone=backbone,num_out=num_out,togray=togray,scramble=scramble,remove_batchnorm=remove_batchnorm,select_kernels=select_kernels)
        #remove last layer from each branch (fc)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor2 = torch.nn.Sequential(*list(self.feature_extractor2.children())[:-1])
        #HEAD (this will be erased in case we do not put --not_remove_existing_head)
        len_fe=len(list(self.feature_extractor.parameters())[-1])
        len_fe2=len(list(self.feature_extractor2.parameters())[-1])
        self.fc=torch.nn.Linear(in_features=len_fe+len_fe2, out_features=num_out, bias=True) #head
    '''
    def forward(self,x1,x2):
        return self.forward_concat(x1,x2)
    '''
    