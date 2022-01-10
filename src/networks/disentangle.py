import torch
import torchvision
import torch.nn as nn

from .extra_layers import *
import pdb
from utils import visualize_batch
__all__ = ['disentangle']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding for the SHAPE BRANCH"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution for the COLOR BRANCH"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_bn_layer=True, is_color=True):
        super(BasicBlock, self).__init__()
        self.use_bn_layer = use_bn_layer
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if is_color:
            self.conv1 = conv1x1(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        if self.use_bn_layer:
            self.bn1 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        if is_color:
            self.conv2 = conv1x1(planes, planes)
        else:
            self.conv2 = conv3x3(planes, planes)
        if self.use_bn_layer:
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.use_bn_layer:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class DisentangleNet(nn.Module):

    def __init__(self, block, layers, design, use_bn=True, factors=[1.0,1.0],smooth_color=False, num_classes=1000, zero_init_residual=False, width_per_group=64):
        super(DisentangleNet, self).__init__()
        self.use_bn = use_bn
        self.factors = factors
        self.smooth_color=smooth_color
        self.inplanes = 64
        self.base_width = width_per_group
        self.head_var = 'fc'
        self.design=design
        # SHAPE BRANCH -- basically same as the network but we make sure the input is Grayscale 1 Channel
        self.conv1_shape = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #if self.use_bn:
        self.bn1_shape = nn.BatchNorm2d(self.inplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_shape = nn.ReLU(inplace=True)
        self.maxpool_shape = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_shape = self._make_layer(block,  64, layers[0], use_bn_layer=self.use_bn, is_color=False)
        self.layer2_shape = self._make_layer(block, 128, layers[1], stride=2, use_bn_layer=self.use_bn, is_color=False)
        self.layer3_shape = self._make_layer(block, 256, layers[2], stride=2, use_bn_layer=self.use_bn, is_color=False)
        self.layer4_shape = self._make_layer(block, 512, layers[3], stride=2, use_bn_layer=self.use_bn, is_color=False)
        self.glayer=GaussianLayer()
        # COLOR BRANCH -- basically same as the network but with 1x1 kernel filters
        if self.smooth_color==True:
            self.conv1_color = nn.Conv2d(6, self.inplanes, kernel_size=1, stride=2, padding=0, bias=False) #6=number of channels (3 color +3 blurred) # padding to 0
        else:
            self.conv1_color = nn.Conv2d(3, self.inplanes, kernel_size=1, stride=2, padding=0, bias=False)
        if self.use_bn:
            self.bn1_color = nn.BatchNorm2d(self.inplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_color = nn.ReLU(inplace=True)
        self.maxpool_color = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_color = self._make_layer(block, 64, layers[0], use_bn_layer=self.use_bn, is_color=True)
        self.layer2_color = self._make_layer(block, 128, layers[1], stride=2, use_bn_layer=self.use_bn, is_color=True)
        self.layer3_color = self._make_layer(block, 256, layers[2], stride=2, use_bn_layer=self.use_bn, is_color=True)
        self.layer4_color = self._make_layer(block, 512, layers[3], stride=2, use_bn_layer=self.use_bn, is_color=True)

        # DIFFERENT OPTIONS TO MERGE THE BRANCHES -- changes for new designs go here
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # -------------------------------------------------------------------------------------------------------------
        if design == 'alternative2pp':
            aa, bb = 60, 100
            self.post_column_color = nn.Sequential(nn.Conv2d(512, aa, kernel_size=1, stride=1, padding=0, bias=False))
            self.post_column_shape = nn.Sequential(nn.Conv2d(512, bb, kernel_size=1, stride=1, padding=0, bias=False))
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(
                nn.Conv2d(aa + bb, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True), self.avgpool, Flatten())
            self.fc = nn.Linear(num_classes, num_classes)
        elif design == 'design1':
            self.post_column_color = nn.Sequential(self.avgpool, Flatten())
            self.post_column_shape = nn.Sequential(self.avgpool, Flatten())
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(nn.Linear((512 + 512) * 1, (512 + 512) * 1))
            self.fc = nn.Linear((512 + 512) * 1, num_classes)
        elif design == 'design1b':
            self.post_column_color = nn.Sequential(self.avgpool, Flatten())
            self.post_column_shape = nn.Sequential(self.avgpool, Flatten())
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(nn.Linear((512 + 512) * 1, (512 + 512) * 1), nn.ReLU(inplace=True))
            self.fc = nn.Linear((512 + 512) * 1, num_classes)
        elif design == 'design2':
            self.post_column_color = nn.Sequential(self.avgpool)
            self.post_column_shape = nn.Sequential(self.avgpool)
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear((512 + 512) * 1, num_classes)
        elif design == 'design3':
            aa, bb = 60, 100
            self.post_column_color = nn.Sequential(nn.Conv2d(512, aa, kernel_size=1, stride=1, padding=0, bias=False))
            self.post_column_shape = nn.Sequential(nn.Conv2d(512, bb, kernel_size=1, stride=1, padding=0, bias=False))
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear((aa + bb) * 7 * 7, num_classes)
        elif design == 'design3b':
            aa, bb = 32, 128
            self.post_column_color = nn.Sequential(nn.Conv2d(512, aa, kernel_size=1, stride=1, padding=0, bias=False))
            self.post_column_shape = nn.Sequential(nn.Conv2d(512, bb, kernel_size=1, stride=1, padding=0, bias=False))
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten(),
                                              nn.Dropout(0.5))
            self.fc = nn.Linear((aa + bb) * 7 * 7, num_classes)
        elif design == 'design3c':
            aa, bb = 32, 128
            self.post_column_color = nn.Sequential(nn.Conv2d(512, aa, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.ReLU(inplace=True))
            self.post_column_shape = nn.Sequential(nn.Conv2d(512, bb, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.ReLU(inplace=True))
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear((aa + bb) * 7 * 7, num_classes)
        elif design == 'design4':
            self.post_column_color = nn.Sequential()
            self.post_column_shape = nn.Sequential()
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten(), nn.Dropout(0.5), nn.Linear((512+512)*7*7, 256))
            self.fc = nn.Linear(256, num_classes)
        elif design == 'design5':  # same as design 3b ?
            aa, bb = 32, 128
            self.post_column_color = nn.Sequential(nn.Conv2d(512, aa, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.ReLU(inplace=True))
            self.post_column_shape = nn.Sequential(nn.Conv2d(512, bb, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.ReLU(inplace=True))
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear((aa + bb) * 7 * 7, num_classes)
        elif design == 'design6':
            aa, bb = 32, 128
            self.post_column_color = nn.Sequential()
            self.post_column_shape = nn.Sequential()
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear((512 + 512) * 7 * 7, num_classes)
        elif design == 'design7':
            aa, bb = 32, 128
            self.post_column_color = nn.Sequential()
            self.post_column_shape = nn.Sequential()
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear((512 + 512) * 7 * 7, num_classes)
        elif design == 'design8':
            aa, bb = 32, 128
            self.post_column_color = nn.Sequential()
            self.post_column_shape = nn.Sequential()
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Sequential()
        elif design == 'shapeonly':
            #replace all post-mechanism by void (sequential)
            self.post_column_color = nn.Sequential()
            self.post_column_shape = nn.Sequential(self.avgpool)
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            #remove all color attributes
            delattr(self,'conv1_color')
            delattr(self,'bn1_color')
            delattr(self,'relu_color')
            delattr(self,'maxpool_color')
            delattr(self,'layer1_color')
            delattr(self,'layer2_color')
            delattr(self,'layer3_color')
            delattr(self,'layer4_color')
        elif design == 'coloronly':
            #replace all post-mechanism by void (sequential)
            self.post_column_color = nn.Sequential(self.avgpool)
            self.post_column_shape = nn.Sequential()
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential(Flatten())
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            #remove all shape attributes
            delattr(self,'conv1_shape')
            delattr(self,'bn1_shape')
            delattr(self,'relu_shape')
            delattr(self,'maxpool_shape')
            delattr(self,'layer1_shape')
            delattr(self,'layer2_shape')
            delattr(self,'layer3_shape')
            delattr(self,'layer4_shape')
        else:
            # Default: VANILLA
            print('Warning: default vanilla design being used.')
            self.post_column_color = nn.Sequential(self.avgpool, Flatten())
            self.post_column_shape = nn.Sequential(self.avgpool, Flatten())
            self.column_merging = Concatenate()
            self.post_merging = nn.Sequential()
            self.fc = nn.Linear((512 + 512) * block.expansion, num_classes)
        # -------------------------------------------------------------------------------------------------------------

        # INITIALIZATION STUFF
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # THIS MANAGES THE GENERATION OF THE SHAPE AND COLOR COLUMNS
    def _make_layer(self, block, planes, blocks, stride=1, use_bn_layer=True, is_color=True):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn_layer:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes*block.expansion, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                )
        layers.append(block(self.inplanes, planes, stride, downsample, use_bn_layer=use_bn_layer, is_color=is_color))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_bn_layer=use_bn_layer, is_color=is_color))
        return nn.Sequential(*layers)

    # FOR SOME REASON, ORIGINAL CODE MADE THIS TWO FORWARD FUNCTIONS, SO I KEEP THEM -- See note [TorchScript super()]
    def _forward_impl(self, x):
        if self.smooth_color == True:
            gx=self.glayer(x) #create blurred images
            x_color = self.column_merging([x,gx]) #concat blurred images
        else:
            x_color = x
        x_shape = (torch.mean(x, dim=1, keepdim=True))  #.repeat(1, 3, 1, 1)  # no need replicate grayscale to channels
        if self.design=="shapeonly":
            if self.use_bn:
                x_shape = self.maxpool_shape(self.relu_shape(self.bn1_shape(self.conv1_shape(x_shape))))
            else:
                x_shape = self.maxpool_shape(self.relu_shape(self.conv1_shape(x_shape)))
            x_shape = self.layer4_shape(self.layer3_shape(self.layer2_shape(self.layer1_shape(x_shape))))
            x_shape = self.post_column_shape(x_shape)
            x = self.post_merging(x_shape)
        elif self.design=="coloronly":
            if self.use_bn:
                x_color = self.maxpool_color(self.relu_color(self.bn1_color(self.conv1_color(x_color))))
            else:
                x_color = self.maxpool_color(self.relu_color(self.conv1_color(x_color)))
            x_color = self.layer4_color(self.layer3_color(self.layer2_color(self.layer1_color(x_color))))
            x_color = self.post_column_color(x_color)
            xx = x.clone()
            x = self.post_merging(x_color)
        else:
            if self.use_bn:
                x_color = self.maxpool_color(self.relu_color(self.bn1_color(self.conv1_color(x_color))))
                x_shape = self.maxpool_shape(self.relu_shape(self.bn1_shape(self.conv1_shape(x_shape))))
            else:
                x_color = self.maxpool_color(self.relu_color(self.conv1_color(x_color)))
                x_shape = self.maxpool_shape(self.relu_shape(self.bn1_shape(self.conv1_shape(x_shape)))) #shape (3x3|7x7) must have bn
            x_color = self.layer4_color(self.layer3_color(self.layer2_color(self.layer1_color(x_color))))
            x_shape = self.layer4_shape(self.layer3_shape(self.layer2_shape(self.layer1_shape(x_shape))))

            #multiplicative factors
            x_color=x_color * self.factors[0]
            x_shape=x_shape * self.factors[1]

            # MAIN STRUCTURE TO MERGE THE BRANCHES -- no changes should be made here
            # -------------------------------------------------------------------------------------------------------------
            x_color = self.post_column_color(x_color)
            x_shape = self.post_column_shape(x_shape)
            x = self.column_merging([x_color, x_shape])
            xx = x.clone()
            x = self.post_merging(x)
            # -------------------------------------------------------------------------------------------------------------
        x = self.fc(x)
        
        return x,xx

    def forward(self, x):
        return self._forward_impl(x)


# MAIN FUNCTION TO CALL THE CLASS GENERATING THE SPECIFIC RESNET -- we cannot have pretrained but we keep the argument
def disentangle(pretrained=False, design='vanilla', **kwargs):
    # Right now focused only on working for ResNet-18 --> {BasicBlock, n=2, layers=4}
    model = DisentangleNet(BasicBlock, [2, 2, 2, 2], design=design, **kwargs)
    return model

