from networks.twobranch import twobranch
import torch
import utils
from copy import deepcopy

class twobranch_concat2(twobranch):
    def forward(self,x):
        x1=deepcopy(x)
        if self.scramble is True:
            x1=utils.batch2scramble(deepcopy(x))
        x2=utils.batch2gray(deepcopy(x),transform_type=self.togray)
        return self.forward_concat(x1,x2)
    def forward_concat(self,x1,x2):
        features = self.feature_extractor(x1)
        features = features.view(features.size(0),-1) #flatten
        features2 = self.feature_extractor2(x2)
        features2 = features2.view(features2.size(0),-1) #flatten
        grouped = torch.cat((features,features2),dim=0)
        output = grouped.view(grouped.size(0),-1) #flatten
        output = self.fc(grouped)
        return output
