from networks.twobranch_sum import twobranch_sum
import torch

class twobranch_sum_noFC(twobranch_sum):
    def __init__(self, pretrained=False, backbone='resnet18', num_out=100, togray="mean",scramble=False,select_kernels='all', remove_batchnorm=None):
        self.backbone = backbone
        self.num_out = num_out
        self.togray = togray
        self.scramble = scramble
        self.remove_batchnorm = remove_batchnorm
        self.select_kernels=select_kernels
        #Module init
        super(twobranch_sum_noFC, self).__init__(pretrained=pretrained,backbone=backbone,num_out=num_out,togray=togray,scramble=scramble,remove_batchnorm=remove_batchnorm,select_kernels=select_kernels)
        #remove last layer from each branch (fc)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor2 = torch.nn.Sequential(*list(self.feature_extractor2.children())[:-1])
        #HEAD
        len_fe=len(list(self.feature_extractor.parameters())[-1])
        self.fc=torch.nn.Linear(in_features=len_fe, out_features=self.num_out, bias=True)
        self.head_var = 'fc'