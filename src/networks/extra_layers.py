import torch
import torch.nn as nn


def get_new_head_architecture(out_size, num_cls, head_arch):
    #heads
    if head_arch == 'conv_pool_flat':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, num_cls, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten())
    elif head_arch == 'design5':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, num_cls, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=True),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten())
    elif head_arch == 'design6':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 256, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.Conv2d(256, num_cls, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten())
    elif head_arch == 'design6b':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 512, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.Conv2d(512, num_cls, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten())
    elif head_arch == 'design6c':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.Conv2d(1024, num_cls, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten())
    elif head_arch == 'design6d':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 128, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.Conv2d(128, num_cls, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten())
    elif head_arch == 'design6e': #no relu
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 256, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.Conv2d(256, num_cls, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten())
    elif head_arch == 'design7':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 256, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten(),
                             nn.Linear(256, num_cls),
                             Flatten())
    elif head_arch == 'design7b':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 512, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten(),
                             nn.Linear(512, num_cls),
                             Flatten())
    elif head_arch == 'design7c':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten(),
                             nn.Linear(1024, num_cls),
                             Flatten())
    elif head_arch == 'design7d':
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 128, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.ReLU(inplace=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten(),
                             nn.Linear(128, num_cls),
                             Flatten())
    elif head_arch == 'design7e': #no relu
        return nn.Sequential(Expand_ResNet(out_size), 
                             nn.Conv2d(out_size, 256, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             Flatten(),
                             nn.Linear(256, num_cls),
                             Flatten())
    elif head_arch == 'None':
        return nn.Sequential()
    else:
        # Default: vanilla Linear layer
        return nn.Linear(out_size, num_cls)


class Flatten(torch.nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1)

class Expand_ResNet(torch.nn.Module):
    def __init__(self, out_size=1024):
        super(Expand_ResNet, self).__init__()
        self.out_size=out_size
    def forward(self, input):
        return torch.reshape(input[0], [input[0].size(0), self.out_size,7,7])

class Concatenate(torch.nn.Module):
    def forward(self, input):
        return torch.cat(input, dim=1)

import numpy as np
import scipy.ndimage
class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(10), 
            nn.Conv2d(3, 3, 21, stride=1, padding=0, bias=None, groups=3)
        )

        self.weights_init()
    def forward(self, x):
        with torch.no_grad():
            return self.seq(x)

    def weights_init(self):
        n= np.zeros((21,21))
        n[10,10] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=3)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


