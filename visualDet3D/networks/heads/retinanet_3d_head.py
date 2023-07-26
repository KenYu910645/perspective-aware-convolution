import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.heads.bev_ank_head import BevAnk3DHead
from visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead

class RetinaNet3DHead_BevAnk(BevAnk3DHead):
    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):
        
        # SHARE WEIGHT
        # Classification Branch
        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, 4*num_cls_output, kernel_size=3, padding=1),
            AnchorFlatten(num_cls_output)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        # Regression Branch # Without LookGround
        self.reg_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(),
            nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(reg_feature_size, 4*num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten(num_reg_output)
        )
        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)
        
    def forward(self, input):
        # feats = dict(features=features, P2=P2, image=img_batch)
        # print(f"feats[0].shape = {input['features'][0].shape}") # torch.Size([8, 1024, 36, 160]) 1/8
        # print(f"feats[1].shape = {input['features'][1].shape}") # torch.Size([8, 1024, 18, 80]) 1/16
        # print(f"feats[2].shape = {input['features'][2].shape}") # torch.Size([8, 1024, 9, 40]) 1/32
        # print(f"feats[3].shape = {input['features'][3].shape}") # torch.Size([8, 1024, 5, 20]) 1/64?
        # print(f"feats[4].shape = {input['features'][4].shape}") # torch.Size([8, 1024, 3, 10]) 1/128?

        cls_scores = []
        reg_preds  = []
        for i_feat, feat in enumerate(input['features']):
            cls_feat = self.cls_feature_extraction(feat)
            reg_feat = self.reg_feature_extraction(feat)
            
            cls_scores.append(cls_feat)
            reg_preds.append (reg_feat)
        
        # print(f"reg_preds[0].shape = {reg_preds[0].shape}") # torch.Size([8, 23040, 12])
        # print(f"reg_preds[1].shape = {reg_preds[1].shape}") # torch.Size([8, 5760, 12])
        # print(f"reg_preds[2].shape = {reg_preds[2].shape}") # torch.Size([8, 1440, 12])
        # print(f"reg_preds[3].shape = {reg_preds[3].shape}") # torch.Size([8, 400, 12])
        # print(f"reg_preds[4].shape = {reg_preds[4].shape}") # torch.Size([8, 120, 12])
        cls_scores = torch.cat(cls_scores, dim=1) # [B, N, num_class]
        reg_preds  = torch.cat(reg_preds,  dim=1) # [B, N, 4]

        # print(f"cls_scores = {cls_scores.shape}") # torch.Size([8, 30760, 2])
        # print(f"reg_preds = {reg_preds.shape}") # torch.Size([8, 30760, 12])
        return cls_scores, reg_preds

class RetinaNet3DHead_GACAnk(AnchorBasedDetection3DHead):
    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):
        
        if not self.is_seperate_fpn_level_head : 
            # SHARE WEIGHT
            # Classification Branch
            self.cls_feature_extraction = nn.Sequential(
                nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(cls_feature_size, num_anchors*num_cls_output, kernel_size=3, padding=1),
                AnchorFlatten(num_cls_output)
            )
            self.cls_feature_extraction[-2].weight.data.fill_(0)
            self.cls_feature_extraction[-2].bias.data.fill_(0)

            # Regression Branch # Without LookGround
            self.reg_feature_extraction = nn.Sequential(
                nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(),
                nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
                AnchorFlatten(num_reg_output)
            )
            self.reg_feature_extraction[-2].weight.data.fill_(0)
            self.reg_feature_extraction[-2].bias.data.fill_(0)
            
            print(f"[retinanet_3d_head.py] self.cls_feature_extraction = {self.cls_feature_extraction}")
            print(f"[retinanet_3d_head.py] self.reg_feature_extraction = {self.reg_feature_extraction}")
        else:
            if not self.is_seperate_2d:
                # Non share weight
                self.cls_feature_extraction = nn.ModuleList( [ nn.Sequential(
                    nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
                    nn.Dropout2d(0.3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
                    nn.Dropout2d(0.3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cls_feature_size, num_anchors*num_cls_output, kernel_size=3, padding=1),
                    AnchorFlatten(num_cls_output) ) for _ in range(5) ] )
                for i in range(5):
                    self.cls_feature_extraction[i][-2].weight.data.fill_(0)
                    self.cls_feature_extraction[i][-2].bias.data.fill_(0)
                
                self.reg_feature_extraction = nn.ModuleList( [ nn.Sequential(
                    nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
                    nn.BatchNorm2d(reg_feature_size),
                    nn.ReLU(),
                    nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
                    nn.BatchNorm2d(reg_feature_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
                    AnchorFlatten(num_reg_output) ) for _ in range(5) ] )
                for i in range(5):
                    self.reg_feature_extraction[i][-2].weight.data.fill_(0)
                    self.reg_feature_extraction[i][-2].bias.data.fill_(0)
            else:
                # share weight
                self.cls_feature_extraction = nn.Sequential(
                    nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
                    nn.Dropout2d(0.3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
                    nn.Dropout2d(0.3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cls_feature_size, num_anchors*num_cls_output, kernel_size=3, padding=1),
                    AnchorFlatten(num_cls_output) )
                self.cls_feature_extraction[-2].weight.data.fill_(0)
                self.cls_feature_extraction[-2].bias.data.fill_(0)
                
                # share weight
                self.feature_2d_extraction = nn.Sequential(
                    nn.Conv2d(num_features_in, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256            , 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256            , num_anchors*4, kernel_size=3, padding=1),
                    AnchorFlatten(4),
                )
                self.feature_2d_extraction[-2].weight.data.fill_(0)
                self.feature_2d_extraction[-2].bias.data.fill_(0)
                
                self.reg_feature_extraction = nn.ModuleList( [ nn.Sequential(
                    nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
                    nn.BatchNorm2d(reg_feature_size),
                    nn.ReLU(),
                    nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
                    nn.BatchNorm2d(reg_feature_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reg_feature_size, num_anchors*(num_reg_output-4), kernel_size=3, padding=1),
                    AnchorFlatten(num_reg_output-4),) for _ in range(5) ] )
                for i in range(5):
                    self.reg_feature_extraction[i][-2].weight.data.fill_(0)
                    self.reg_feature_extraction[i][-2].bias.data.fill_(0)
            
    def forward(self, input):
        # feats = dict(features=features, P2=P2, image=img_batch)
        # print(f"feats[0].shape = {input['features'][0].shape}") # torch.Size([8, 1024, 36, 160]) 1/8
        # print(f"feats[1].shape = {input['features'][1].shape}") # torch.Size([8, 1024, 18, 80]) 1/16
        # print(f"feats[2].shape = {input['features'][2].shape}") # torch.Size([8, 1024, 9, 40]) 1/32
        # print(f"feats[3].shape = {input['features'][3].shape}") # torch.Size([8, 1024, 5, 20]) 1/64?
        # print(f"feats[4].shape = {input['features'][4].shape}") # torch.Size([8, 1024, 3, 10]) 1/128?
        
        cls_scores = []
        reg_preds  = []
        preds_2d    = [] # for is_seperate_2d
        for i_feat, feat in enumerate(input['features']):
            
            if self.is_seperate_fpn_level_head:
                if self.is_seperate_2d:
                    cls_feat = self.cls_feature_extraction(feat)
                    feat_2d  = self.feature_2d_extraction(feat)
                    reg_feat = self.reg_feature_extraction[i_feat](feat)
                else:
                    cls_feat = self.cls_feature_extraction[i_feat](feat)
                    reg_feat = self.reg_feature_extraction[i_feat](feat)
            else:
                cls_feat = self.cls_feature_extraction(feat)
                reg_feat = self.reg_feature_extraction(feat)
            cls_scores.append(cls_feat)
            reg_preds.append (reg_feat)
            
            if self.is_seperate_2d:
                preds_2d.append(feat_2d)
        
        # print(f"reg_preds[0].shape = {reg_preds[0].shape}") # torch.Size([8, 23040, 12])
        # print(f"reg_preds[1].shape = {reg_preds[1].shape}") # torch.Size([8, 5760, 12])
        # print(f"reg_preds[2].shape = {reg_preds[2].shape}") # torch.Size([8, 1440, 12])
        # print(f"reg_preds[3].shape = {reg_preds[3].shape}") # torch.Size([8, 400, 12])
        # print(f"reg_preds[4].shape = {reg_preds[4].shape}") # torch.Size([8, 120, 12])
        
        cls_scores = torch.cat(cls_scores, dim=1) # [B, N, num_class]
        reg_preds  = torch.cat(reg_preds,  dim=1) # [B, N, 12]
        if self.is_seperate_2d:
            preds_2d  = torch.cat(preds_2d,  dim=1) # [B, N, 4]

            # print(f"cls_scores = {cls_scores.shape}") # torch.Size([8, 69210, 2])
            # print(f"preds_2d = {preds_2d.shape}") # torch.Size([8, 69210, 4])
            # print(f"reg_preds = {reg_preds.shape}") # torch.Size([8, 69210, 8])
            
            # Concat seperate value together
            reg_preds = torch.cat([preds_2d , reg_preds], dim=2)

        return cls_scores, reg_preds
