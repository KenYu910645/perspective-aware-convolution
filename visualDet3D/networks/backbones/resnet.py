from typing import Tuple, List, Union
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from visualDet3D.networks.lib.bam import BAM

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    '''
    reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
            (1): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            ) 
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    planes = [64, 128, 256, 512]
    def __init__(self, block:Union[BasicBlock, Bottleneck],
                       layers:Tuple[int, ...],
                       num_stages:int=4,
                       strides:Tuple[int, ...]=(1, 2, 2, 2),
                       dilations:Tuple[int, ...]=(1, 1, 1, 1),
                       out_indices:Tuple[int, ...]=(-1, 0, 1, 2, 3),
                       frozen_stages:int=-1,
                       norm_eval:bool=True,
                       use_bam_in_resnet:bool=False,
                       drop_last_downsample:bool=False,
                       ):
        self.use_bam_in_resnet = use_bam_in_resnet
        self.drop_last_downsample = drop_last_downsample
        print(f"self.use_bam_in_resnet = {self.use_bam_in_resnet}")
        print(f"self.drop_last_downsample = {self.drop_last_downsample}")
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        assert max(out_indices) < num_stages

        self.conv1      = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.coordconv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False) # This is for exp B
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i in range(num_stages):
            setattr(self, f"layer{i+1}", self._make_layer(block, self.planes[i], layers[i], stride=self.strides[i], dilation=self.dilations[i]))
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #prior = 0.01
        if self.use_bam_in_resnet:
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        
        # print(f"self.layer1 = {self.layer1}")
        # print(f"self.layer2 = {self.layer2}")
        # print(f"self.layer3 = {self.layer3}")
        # print(f"self.layer4 = {self.layer4}")
        
        self.train()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(ResNet, self).train(mode)

        if mode:
            self.freeze_stages()
            if self.norm_eval:
                self.freeze_bn()

    def freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
            
            for param in self.bn1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm): # Will freeze both batchnorm and sync batchnorm
                layer.eval()

    def forward(self, img_batch):
        outs = []
        
        #############
        ### Conv1 ###
        #############
        x = self.conv1(img_batch) # Original, 3 channels
        x = self.bn1(x)
        x = self.relu(x)
        
        ###############
        ### Conv2_x ###
        ###############
        if -1 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)
        
        # self.layer1 = conv2_x (3  layers) -> 1/4
        # self.layer2 = conv3_x (4  layers) -> 1/8
        # self.layer3 = conv4_x (23 layers) -> 1/16
        # self.layer4 = conv5_x (3  layers) -> 1/32
        
        # print(f"self.num_stages = {self.num_stages}") # self.num_stages = 3
        for i in range(self.num_stages):
            layer = getattr(self, f"layer{i+1}")
            x = layer(x)
            
            if self.use_bam_in_resnet:
                if   i == 0: x = self.bam1(x)
                elif i == 1: x = self.bam2(x)
                elif i == 2: x = self.bam3(x)
            
            if i in self.out_indices:
                outs.append(x)
            # print(f"x = {x.shape}")
            
            # Resnet34
            # x = torch.Size([8, 64, 72, 320])
            # x = torch.Size([8, 128, 36, 160])
            # x = torch.Size([8, 256, 18, 80])
            # ResNet50
            # x = torch.Size([8, 256, 72, 320])
            # x = torch.Size([8, 512, 36, 160])
            # x = torch.Size([8, 1024, 18, 80])
        return outs


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model

def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model

def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model

def resnet(depth, **kwargs):
    if depth == 18:
        model = resnet18(**kwargs)
    elif depth == 34:
        model = resnet34(**kwargs)
    elif depth == 50:
        model = resnet50(**kwargs)
    elif depth == 101:
        model = resnet101(**kwargs)
    elif depth == 152:
        model = resnet152(**kwargs)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    return model

if __name__ == '__main__':
    model = resnet18(False).cuda()
    model.eval()
    image = torch.rand(2, 3, 224, 224).cuda()
    
    output = model(image)
