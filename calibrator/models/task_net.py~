import torch
import torch.nn as nn
from torch.nn import init
from .models import register_model 
from .util import init_weights
import numpy as np
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,cfg=None):
        super(BasicBlock, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,cfg=None, num_classes=10, divided_by=1):
        super(ResNet, self).__init__()
        self.in_planes = 64//divided_by
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256//divided_by, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512//divided_by, num_blocks[3], stride=2)
        self.linear = nn.Linear(512//divided_by*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,cfg=self.cfg))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None, **kwargs):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def forward(self, x, with_ft=False):
        x = self.conv_params(x)
        ft = x.view(x.size(0), -1)
        x = self.fc_params(ft)
        score = self.classifier(x)
        if with_ft:
            return score, x
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

@register_model('LeNet')
class LeNet(TaskNet):
    "Network used for MNIST or USPS experiments."    

    num_channels = 3
    image_size = 32
    name = 'LeNet'
    out_dim = 500 # dim of last feature layer

    def setup_net(self):

        self.conv_params = nn.Sequential(
                nn.Conv2d(self.num_channels, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.fc_params = nn.Linear(1250, 500)
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(500, self.num_cls)
                )



        
@register_model('DTN')
class DTNClassifier(TaskNet):
    "Classifier used for SVHN->MNIST Experiment"

    num_channels = 3
    image_size = 32
    name = 'DTN'
    out_dim = 512 # dim of last feature layer

    def setup_net(self):
        self.conv_params = nn.Sequential (
                nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                )

        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, self.num_cls)
                )


@register_model('DTN_resnet18')
class ResNetClassifier(ResNet):
    def __init__(self,num_cls=10, weights_init=None):
        super(ResNetClassifier,self).__init__(BasicBlock,[2,2,2,2])
        self.criterion = nn.CrossEntropyLoss()        
        self.num_channels = 3
        self.image_size = 32
        self.name = 'DTN_resnet18'        
    
    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)
@register_model('DTN_w2')
class DTNClassifier(TaskNet):
    "Classifier used for SVHN->MNIST Experiment"

    num_channels = 3
    image_size = 32
    name = 'DTN_w2'
    out_dim = 512 # dim of last feature layer

    def setup_net(self):
        w = 2
        self.conv_params = nn.Sequential (
                nn.Conv2d(self.num_channels, 64*w, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64*w),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64*w, 128*w, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128*w),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128*w, 256*w, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256*w),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4*w, 512*w),
                nn.BatchNorm1d(512*w),
                )

        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512*w, self.num_cls)
                )        
@register_model('DTN_w4')
class DTNClassifier(TaskNet):
    "Classifier used for SVHN->MNIST Experiment"

    num_channels = 3
    image_size = 32
    name = 'DTN'
    out_dim = 512 # dim of last feature layer

    def setup_net(self):
        w = 4        
        self.conv_params = nn.Sequential (
                nn.Conv2d(self.num_channels, 64*w, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64*w),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64*w, 128*w, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128*w),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128*w, 256*w, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256*w),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4*w, 512*w),
                nn.BatchNorm1d(512*w),
                )

        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512*w, self.num_cls)
                )        
@register_model('StrongDTN')
class StrongDTNClassifier(TaskNet):
    "Classifier used for SVHN->MNIST Experiment"

    num_channels = 3
    image_size = 32
    name = 'StrongDTN'
    

    def setup_net(self):
        norm_layer = nn.BatchNorm2d
        ndf = 64        
        kw = 3
        sequence = [
            nn.Conv2d(3, ndf, kernel_size=kw, stride=2),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]
        self.conv_params = nn.Sequential(*sequence)
        self.fc_params = nn.Sequential(nn.Linear(ndf*nf_mult,1024))
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 10)
        )


