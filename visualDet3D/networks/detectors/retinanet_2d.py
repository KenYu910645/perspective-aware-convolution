import numpy as np
import torch.nn as nn
import torch
import math
import time
from torchvision.ops import nms
from visualDet3D.networks.utils import DETECTOR_DICT
from visualDet3D.networks.utils.utils import BBoxTransform, ClipBoxes
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.heads.retinanet_head import RetinanetHead
from visualDet3D.networks.heads import losses
from visualDet3D.networks.lib.blocks import ConvBnReLU
from visualDet3D.networks.backbones import resnet

class FPN(nn.Module):
    """Some Information about FPN"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_outs,
                 is_use_final_conv = True,
                 is_use_latenal_connection = True):
        super(FPN, self).__init__()
        self.in_channels = in_channels # [512, 1024, 2048]
        self.is_use_final_conv = is_use_final_conv
        self.is_use_latenal_connection = is_use_latenal_connection
        print(f"is_use_final_conv = {self.is_use_final_conv}")
        print(f"is_use_latenal_connection = {self.is_use_latenal_connection}")
        
        '''
        (lateral_convs): ModuleList(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
        '''
        # 1x1 convolution
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels[i], out_channels, 1) for i in range(len(in_channels))
            ]
        )
        '''
        (fpn_convs): ModuleList(
            (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        '''
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, (3, 3), padding=1) for i in range(len(in_channels))
            ]
        )

        extra_levels = num_outs - len(in_channels) # 2 # add extra down-sampling, Retinanet add convs on inputs
        if extra_levels > 0: # 2
            for i in range(extra_levels):
                if i == 0:
                    self.fpn_convs.append(nn.Conv2d(in_channels[-1], out_channels, 3, padding=1, stride=2))
                else:
                    self.fpn_convs.append(nn.Conv2d(out_channels   , out_channels, 3, padding=1, stride=2))
        # self.conv_debug_8 = nn.Conv2d(512, 1024, 3, padding=1)
        # self.conv_debug_16 = nn.Conv2d(1024, 1024, 3, padding=1)
        # self.conv_debug_32 = nn.Conv2d(2048, 1024, 3, padding=1)
        # k = torch.tensor([[0.0, 0.0, 0.0,], [0.0, 1.0/1024.0, 0.0], [0.0, 0.0, 0.0]])
        # self.conv_debug_16.weight.data = torch.tile(k, (1024, 1024, 1, 1)).to("cuda")
        # print(self.conv_debug_16.weight)
        # self.conv_debug_16.bias.data.fill_(0)


    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # self.in_channels = [512, 1024, 2048]
        # print(f"feats[0].shape = {feats[0].shape}") # torch.Size([8, 512, 36, 160])
        # print(f"feats[1].shape = {feats[1].shape}") # torch.Size([8, 1024, 18, 80])
        # print(f"feats[2].shape = {feats[2].shape}") # torch.Size([8, 2048, 9, 40])

        # Build Laterals, 1x1 convolution
        outs = [ self.lateral_convs[i](feats[i]) for i in range(len(self.in_channels)) ]
        # print(f"laterals[0].shape = {laterals[0].shape}") # torch.Size([8, 1024, 36, 160])
        # print(f"laterals[1].shape = {laterals[1].shape}") # torch.Size([8, 1024, 18, 80])
        # print(f"laterals[2].shape = {laterals[2].shape}") # torch.Size([8, 1024, 9, 40])

        # top-down path, element-wise addition
        if self.is_use_latenal_connection:
            for i in range(len(self.in_channels) - 1, 0, -1): # i = 2, 1
                outs[i - 1] += torch.nn.functional.interpolate(outs[i], scale_factor=2, mode='nearest')
        
        # build original levels
        if self.is_use_final_conv:
            outs = [ self.fpn_convs[i](outs[i]) for i in range(len(self.in_channels)) ]
        # print(f"outs[0].shape = {outs[0].shape}") # torch.Size([8, 1024, 36, 160])
        # print(f"outs[1].shape = {outs[1].shape}") # torch.Size([8, 1024, 18, 80])
        # print(f"outs[2].shape = {outs[2].shape}") # torch.Size([8, 1024, 9, 40])
        # print("========================================")
        # print(feats[1][0, 0, :, :])
        # outs = [self.conv_debug_8(feats[0]),
        #         self.conv_debug_16(feats[1]),
        #         self.conv_debug_32(feats[2])]
        # print(outs[1][0, 0, :, :])
        # print(self.conv_debug_16.weight.shape) # torch.Size([1024, 1024, 3, 3])
        # print(self.conv_debug_16.weight.mean()) # torch.Size([1024])
        
        # Add extra layer
        if len(self.fpn_convs) > len(outs):
            # RetinaNet add convolutions to inputs
            outs.append( self.fpn_convs[len(outs)](feats[-1]) )

            for i in range(len(outs), len(self.fpn_convs)):
                outs.append(self.fpn_convs[i](outs[-1])) # default no relu in mmdetection retinanet, with relu in pytorch/retinanet
        
        # print(f"outs[0].shape = {outs[0].shape}") # torch.Size([8, 1024, 36, 160])
        # print(f"outs[1].shape = {outs[1].shape}") # torch.Size([8, 1024, 18, 80])
        # print(f"outs[2].shape = {outs[2].shape}") # torch.Size([8, 1024, 9, 40])
        # print(f"outs[3].shape = {outs[3].shape}") # torch.Size([8, 1024, 5, 20])
        # print(f"outs[4].shape = {outs[4].shape}") # torch.Size([8, 1024, 3, 10])

        return tuple(outs) # len(out) == 5

class RetinaNetCore(nn.Module):
    """Some Information about RetinaNetCore"""
    def __init__(self, backbone_cfg, neck_cfg):
        super(RetinaNetCore, self).__init__()
        self.backbone = resnet(**backbone_cfg)
        self.neck     = FPN(**neck_cfg)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return feats

class RetinaNet3DCore(nn.Module):
    """Some Information about RetinaNet3DCore"""
    def __init__(self, backbone_cfg, neck_cfg):
        super(RetinaNet3DCore, self).__init__()
        self.backbone = resnet(**backbone_cfg)
        self.neck     = FPN(**neck_cfg)
        print(self.neck)
        
    def forward(self, x):
        feats = self.backbone(x['image'])
        # print(f"feats[0] = {feats[0].shape}") # torch.Size([8, 512, 36, 160])
        # print(f"feats[1] = {feats[1].shape}") # torch.Size([8, 1024, 18, 80])
        # print(f"feats[2] = {feats[2].shape}") # torch.Size([8, 2048, 9, 40])
        feats = self.neck(feats)
        return feats

@DETECTOR_DICT.register_module
class RetinaNet(nn.Module):
    """
        RetinaNet for 2D object detection. Learning from mmdetection but fit in our resign.
    """
    def __init__(self, network_cfg):
        super(RetinaNet, self).__init__()

        self.clipBoxes = ClipBoxes()

        self.obj_types = network_cfg.obj_types

        self.build_core(network_cfg)

        self.build_head(network_cfg)

        self.network_cfg = network_cfg
        
        self.is_writen_anchor_file = False
        
        self.anchor_name = network_cfg.head.anchor_name

    def build_core(self, network_cfg):
        self.core = RetinaNetCore(network_cfg.backbone, network_cfg.neck)

    def build_head(self, network_cfg):
        self.bbox_head = RetinanetHead(**(network_cfg.head) )


    def training_forward(self, img_batch, annotations):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        # print(f"annotations.shape = {annotations.shape}") # torch.Size([8, 7, 16])
        # print(f"img_batch.shape = {img_batch.shape}") # torch.Size([8, 3, 288, 1280])
        
        feats = self.core(img_batch)
        # print(f"feats[0].shape = {feats[0].shape}") # torch.Size([8, 256, 36, 160]) 1/8
        # print(f"feats[1].shape = {feats[1].shape}") # torch.Size([8, 256, 18, 80]) 1/16
        # print(f"feats[2].shape = {feats[2].shape}") # torch.Size([8, 256, 9, 40]) 1/32
        # print(f"feats[3].shape = {feats[3].shape}") # torch.Size([8, 256, 5, 20]) 1/64?
        # print(f"feats[4].shape = {feats[4].shape}") # torch.Size([8, 256, 3, 10]) 1/128?
        # 7690 = 36*160 + 18*80 + 9*40 + 5*20 + 3*10
        # 7690 = 5760   + 1440  + 360  + 100  + 30
        # 7690*9 = 51840  + 12960 + 3240 + 900  + 270
        # 69210  = 51840  + 12960 + 3240 + 900  + 270
        
        cls_preds, reg_preds = self.bbox_head(feats)
        # print(f"cls_preds.shape = {cls_preds.shape}") # torch.Size([8, 69210, 3]) # Three classes
        # print(f"reg_preds.shape = {reg_preds.shape}") # torch.Size([8, 69210, 4])
        
        anchors = self.bbox_head.get_anchor(img_batch)
        # print(f"anchors = {anchors.shape}") # torch.Size([1, 69210, 4])
                
        # Output GAC-implemented anchor for debugging
        # if not self.is_writen_anchor_file:
        #     import pickle
        #     with open("retinanet_2d_anchor.pkl", 'wb') as f:
        #         pickle.dump(anchors, f)
        #         print(f"Save to retinanet_2d_anchor.pkl")
        #     self.is_writen_anchor_file = True

        loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations)
        return loss_dict
    
    def test_forward(self, img_batch):
        """
        Args:
            img_batch: [B, C, H, W] tensor
        Returns:
            results: a nested list:
                result[i] = detection_results for obj_types[i]
                    each detection result is a list [scores, bbox, obj_type]:
                        bbox = [bbox2d(length=4) , cx, cy, z, w, h, l, alpha]
        """
        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing

        feats = self.core(img_batch)
        cls_preds, reg_preds = self.bbox_head(feats)
        anchors = self.bbox_head.get_anchor(img_batch)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, calib = inputs
            return self.training_forward(img_batch, annotations)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch)