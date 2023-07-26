'''
This file is before noam refactoring
'''
import torch
import torch.nn as nn
from torchvision.ops import nms
from easydict import EasyDict
import numpy as np
from visualDet3D.networks.heads.losses import CIoULoss, DIoULoss, SigmoidFocalLoss, ModifiedSmoothL1Loss, IoULoss
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BackProjection, BBox3dProjector
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.networks.utils.utils import ClipBoxes
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.backbones.resnet import BasicBlock
from visualDet3D.networks.lib.ops import ModulatedDeformConvPack
from visualDet3D.networks.lib.look_ground import LookGround
from visualDet3D.networks.lib.cbam import CBAM
from visualDet3D.networks.lib.bam import BAM
from visualDet3D.networks.lib.coordatten import CoordAtt
from visualDet3D.networks.lib.pac import PerspectiveConv2d
# from visualDet3D.networks.lib.pac_legacy import PerspectiveConv2d
from visualDet3D.networks.lib.dcn import DeformableConv2d

class AnchorBasedDetection3DHead(nn.Module):
    def __init__(self, num_features_in:int=1024,
                       num_classes:int=3,
                       num_regression_loss_terms=12,
                       preprocessed_path:str='',
                       anchors_cfg:EasyDict=EasyDict(),
                       layer_cfg:EasyDict=EasyDict(),
                       loss_cfg:EasyDict=EasyDict(),
                       test_cfg:EasyDict=EasyDict(),
                       read_precompute_anchor:bool=True,
                       exp:str='',
                       data_cfg:EasyDict=EasyDict(),
                       is_fpn_debug:bool=False,
                       use_channel_attention:bool=False,
                       use_spatial_attention:bool=False,
                       is_seperate_cz:bool=False,
                       use_bam:bool=False,
                       is_seperate_fpn_level_head:bool=False,
                       use_coordinate_attetion:bool=False,
                       is_fc_depth:bool=False,
                       cz_reg_dim:int=1024,
                       reg_2d_dim:int=256,
                       num_dcnv2:int=0,
                       num_pac_layer:int=0,
                       pac_mode:str="",
                       offset_3d:float=0.4,
                       offset_2d:int=32,
                       pad_mode:str="constant",
                       adpative_P2:bool=False,
                       cz_pred_mode:str="look_ground",
                       is_noam_loss:bool=False,
                       is_pac_bn_relu:bool=False,
                       is_seperate_noam:bool=True):

        super(AnchorBasedDetection3DHead, self).__init__()
        self.anchors = Anchors(preprocessed_path=preprocessed_path, readConfigFile=read_precompute_anchor, **anchors_cfg)
        
        self.num_classes = num_classes
        self.num_regression_loss_terms=num_regression_loss_terms
        self.decode_before_loss = getattr(loss_cfg, 'decode_before_loss', False)
        self.loss_cfg = loss_cfg
        self.test_cfg  = test_cfg
        self.data_cfg = data_cfg
        self.build_loss(**loss_cfg)
        self.backprojector = BackProjection()
        self.clipper = ClipBoxes()
        
        self.exp = exp
        self.layer_cfg = layer_cfg
        self.is_fpn_debug = is_fpn_debug
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        self.is_seperate_cz = is_seperate_cz
        self.use_bam = use_bam
        self.is_seperate_fpn_level_head = is_seperate_fpn_level_head
        self.use_coordinate_attetion = use_coordinate_attetion
        self.is_fc_depth = is_fc_depth
        self.cz_reg_dim  = cz_reg_dim
        self.reg_2d_dim  = reg_2d_dim
        self.num_dcnv2   = num_dcnv2
        self.num_pac_layer = num_pac_layer
        self.pac_mode = pac_mode
        self.offset_3d = offset_3d
        self.offset_2d = offset_2d
        self.pad_mode = pad_mode
        self.adpative_P2 = adpative_P2
        self.cz_pred_mode = cz_pred_mode
        self.is_noam_loss = is_noam_loss
        self.is_pac_bn_relu = is_pac_bn_relu
        self.is_seperate_noam = is_seperate_noam
        
        print(f"AnchorBasedDetection3DHead self.exp = {exp}")
        print(f"iou_type = {self.loss_cfg.iou_type}")
        print(f"self.is_fpn_debug = {self.is_fpn_debug}")
        print(f"self.use_channel_attention = {self.use_channel_attention}")
        print(f"self.use_spatial_attention = {self.use_spatial_attention}")
        print(f"self.is_seperate_cz = {self.is_seperate_cz}")
        print(f"self.use_bam = {self.use_bam}")
        print(f"self.is_seperate_fpn_level_head = {self.is_seperate_fpn_level_head}")
        print(f"use_coordinate_attetion = {self.use_coordinate_attetion}")
        print(f"is_fc_depth = {self.is_fc_depth}")
        print(f"cz_reg_dim = {self.cz_reg_dim}")
        print(f"reg_2d_dim = {self.reg_2d_dim}")
        print(f"num_dcnv2 = {self.num_dcnv2}")
        print(f"self.num_pac_layer=  {self.num_pac_layer}")
        print(f"self.pac_mode=  {self.pac_mode}")
        print(f"adpative_P2 = {self.adpative_P2}")
        print(f"self.offset_3d=  {self.offset_3d}")
        print(f"self.pad_mode=  {self.pad_mode}")
        print(f"self.cz_pred_mode=  {self.cz_pred_mode}")
        print(f"self.is_noam_loss=  {self.is_noam_loss}")
        print(f"self.offset_2d = {self.offset_2d}")
        print(f"is_pac_bn_relu = {self.is_pac_bn_relu}")
        print(f"is_seperate_noam = {self.is_seperate_noam}")
        
        # print(f"self.anchors.num_anchors = {self.anchors.num_anchors}") # 32
        if getattr(layer_cfg, 'num_anchors', None) is None:
            layer_cfg['num_anchors'] = self.anchors.num_anchors
        self.init_layers(**layer_cfg)

        # For Anchor staticical
        self.n_miss_gt = 0
        self.n_cover_gt = 0
        self.n_assign_anchor = 0

    def init_layers(self, num_features_in, # This function will be overwritten
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):

        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(cls_feature_size, num_anchors*(num_cls_output), kernel_size=3, padding=1),
            AnchorFlatten(num_cls_output)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction = nn.Sequential(
            ModulatedDeformConvPack(num_features_in, reg_feature_size, 3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),

            nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten(num_reg_output)
        )

        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

    def forward(self, inputs): # This function will be overwritten
        cls_preds = self.cls_feature_extraction(inputs['features'])
        reg_preds = self.reg_feature_extraction(inputs['features'])

        return cls_preds, reg_preds
        
    def build_loss(self, focal_loss_gamma=0.0, balance_weight=[0], L1_regression_alpha=9, **kwargs):
        self.focal_loss_gamma = focal_loss_gamma
        self.register_buffer("balance_weights", torch.tensor(balance_weight, dtype=torch.float32))
        self.loss_cls = SigmoidFocalLoss(gamma=focal_loss_gamma, balance_weights=self.balance_weights)
        self.loss_bbox = ModifiedSmoothL1Loss(L1_regression_alpha)

        if self.loss_cfg.iou_type == "iou":
            self.loss_iou = IoULoss()
        elif self.loss_cfg.iou_type == "diou":
            self.loss_iou = DIoULoss()
        elif self.loss_cfg.iou_type == "ciou":
            self.loss_iou = CIoULoss()
        else:
            self.loss_iou = IoULoss()

        regression_weight = kwargs.get("regression_weight", [1 for _ in range(self.num_regression_loss_terms)]) #default 12 only use in 3D
        self.register_buffer("regression_weight", torch.tensor(regression_weight, dtype=torch.float))

        self.alpha_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _assign(self, anchor, annotation, 
                    bg_iou_threshold=0.0,
                    fg_iou_threshold=0.5,
                    min_iou_threshold=0.0,
                    match_low_quality=True,
                    gt_max_assign_all=True,
                    **kwargs):
        """
            DOES NOT USE PREDICTION 
            This function decide which anchors should be assign to ground true and make it a "positive anchor"
            I believe "positive anchor" means that anchor is responsible to predict that ground true
            Note that it use 2D IOU to determine the assignment not by 3D IOU
            This function use max 2d IOU to assign gt to anchor, making some groundtrue unattened......

            According to default config.py, anchor box that IOU with gt is smaller than 0.4 is considered as negative sample
            and anchor's IOU greater than 0.5 are considered as positive

            anchor: [N, 4]
            annotation: [num_gt, 4]:
        """
        N = anchor.shape[0]
        num_gt = annotation.shape[0]
        assigned_gt_inds = anchor.new_full(
            (N, ),
            -1, dtype=torch.long
        ) #[N, ] torch.long
        max_overlaps = anchor.new_zeros((N, ))
        assigned_labels = anchor.new_full((N, ),
            -1,
            dtype=torch.long)

        if num_gt == 0:
            assigned_gt_inds = anchor.new_full(
                (N, ),
                0, dtype=torch.long
            ) #[N, ] torch.long
            return_dict = dict(
                num_gt=num_gt,
                assigned_gt_inds = assigned_gt_inds,
                max_overlaps = max_overlaps,
                labels=assigned_labels
            )
            return return_dict

        IoU = calc_iou(anchor, annotation[:, :4]) # num_anchors x num_annotations
        # print(f"IoU = {IoU.shape}") # [3860, 4]

        # max for anchor
        max_overlaps, argmax_overlaps = IoU.max(dim=1) # num_anchors

        unique, counts = np.unique(argmax_overlaps.cpu().numpy(), return_counts=True)
        # print(dict(zip(unique, counts))) # {0: 2555, 1: 529, 2: 552, 3: 224}
        # print(max_overlaps[ argmax_overlaps == 1 ]  )
        # print(f"argmax_overlaps = {argmax_overlaps.shape}") # [3860]
        # print(f"max_overlaps = {max_overlaps.shape}") # [3860]
        # argmax_overlaps

        # max for gt
        gt_max_overlaps, gt_argmax_overlaps = IoU.max(dim=0) #num_gt

        # print(f"max_overlaps = {max_overlaps.min()}")
        # print(f"bg_iou_threshold = {bg_iou_threshold}") # 0.4 -> define in config.py
        # assign negative
        assigned_gt_inds[(max_overlaps >=0) & (max_overlaps < bg_iou_threshold)] = 0

        # assign positive
        pos_inds = max_overlaps >= fg_iou_threshold
        # print( argmax_overlaps[pos_inds] == 0 )

        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if match_low_quality: # match_low_quality = False in config.py
            for i in range(num_gt):
                if gt_max_overlaps[i] >= min_iou_threshold:
                    if gt_max_assign_all:
                        max_iou_inds = IoU[:, i] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i+1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i+1

        
        assigned_labels = assigned_gt_inds.new_full((N, ), -1)
        pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False
            ).squeeze()
        if pos_inds.numel()>0:
            assigned_labels[pos_inds] = annotation[assigned_gt_inds[pos_inds] - 1, 4].long()

        return_dict = dict(
            num_gt = num_gt,
            assigned_gt_inds = assigned_gt_inds,
            max_overlaps  = max_overlaps,
            labels = assigned_labels
        )
        return return_dict

    def _encode(self, sampled_anchors, sampled_gt_bboxes, selected_anchors_3d):
        assert sampled_anchors.shape[0] == sampled_gt_bboxes.shape[0]
        '''
        This function calcuate difference between anchor and ground true -> encode into format that network should be predicting
        
        # print(f"pos_bboxes = {pos_bboxes.shape}") # [40, 4]
        # print(f"pos_gt_bboxes = {pos_gt_bboxes.shape}") # [40, 12]
        # print(f"selected_anchor_3d = {selected_anchor_3d.shape}") # [40, 6, 2]

        Input: 
            * N_P: Number of postive anchor
            sampled_anchors - [N_P, 4] - [40, 4]
                Anchor's 2D bbox - [x1, y1, x2, y2]
            
            sampled_gt_bboxes - [N_P, 12] - [40, 12]
                Ground true that assign to that anchor
                sampled_gt_bboxes = [x1, y1, x2, y2, cls_index, cx, cy , cz, w, h, l , alpha]
                                     0  1   2   3    4          5   6    7   8  9  10, 11
            
            selected_anchors_3d - [N_P, 6, 2]
                3D geometry of anchors, [..., 0] is mean value, [..., 1] is std
                [cz, sin(alpha*2), cos(alpha*2), w, h , l]
                
        '''

        sampled_anchors = sampled_anchors.float()
        sampled_gt_bboxes = sampled_gt_bboxes.float()
        px = (sampled_anchors[..., 0] + sampled_anchors[..., 2]) * 0.5
        py = (sampled_anchors[..., 1] + sampled_anchors[..., 3]) * 0.5
        pw =  sampled_anchors[..., 2] - sampled_anchors[..., 0]
        ph =  sampled_anchors[..., 3] - sampled_anchors[..., 1]

        # ground true 2D bounding box center = (gx, gy)
        gx = (sampled_gt_bboxes[..., 0] + sampled_gt_bboxes[..., 2]) * 0.5
        gy = (sampled_gt_bboxes[..., 1] + sampled_gt_bboxes[..., 3]) * 0.5
        gw =  sampled_gt_bboxes[..., 2] - sampled_gt_bboxes[..., 0]
        gh =  sampled_gt_bboxes[..., 3] - sampled_gt_bboxes[..., 1]

        # diff of 2D bbox's center and geometry 
        targets_dx = (gx - px) / pw # TODO, YOLO says it's unstable to train
        targets_dy = (gy - py) / ph
        targets_dw = torch.log(gw / pw)
        targets_dh = torch.log(gh / ph)

        # 3D bbox center on image plane, # TODO because gac_original use 2d boudning box as feature location
        targets_cdx = (sampled_gt_bboxes[:, 5] - px) / pw # TODO we should use cx, cy prior
        targets_cdy = (sampled_gt_bboxes[:, 6] - py) / ph

        # 3D bbox center's depth
        targets_cdz = (sampled_gt_bboxes[:, 7] - selected_anchors_3d[:, 0, 0]) / selected_anchors_3d[:, 0, 1]
        
        # 3D bbox orientation 
        targets_cd_sin = (torch.sin(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 1, 0]) / selected_anchors_3d[:, 1, 1]
        targets_cd_cos = (torch.cos(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 2, 0]) / selected_anchors_3d[:, 2, 1]
        
        # 3D bbox geometry
        targets_w3d = (sampled_gt_bboxes[:, 8]  - selected_anchors_3d[:, 3, 0]) / selected_anchors_3d[:, 3, 1]
        targets_h3d = (sampled_gt_bboxes[:, 9]  - selected_anchors_3d[:, 4, 0]) / selected_anchors_3d[:, 4, 1]
        targets_l3d = (sampled_gt_bboxes[:, 10] - selected_anchors_3d[:, 5, 0]) / selected_anchors_3d[:, 5, 1]

        # This target values are the values that the network going to predict
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, 
                         targets_cdx, targets_cdy, targets_cdz,
                         targets_cd_sin, targets_cd_cos,
                         targets_w3d, targets_h3d, targets_l3d), dim=1) # 12

        stds = targets.new([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1])

        targets = targets.div_(stds)

        targets_alpha_cls = (torch.cos(sampled_gt_bboxes[:, 11:12]) > 0).float()
        return targets, targets_alpha_cls #[N, 4]

    def _decode(self, boxes, deltas, anchors_3d_mean_std, label_index, alpha_score, cz_deltas = None):
        '''
        Input: 
            * N_D: number of detection 
            boxes - [N_D, 4] - [184, 4]
                anchor of these detection corresspond to 
            deltas - [N_D, 12] - [184, 12]
                prediction result
            anchors_3d_mean_std  - [N_D, 12] - [184, 1, 6, 2]
            label_index - [N_D] - [184]
                prediction result of category
            alpha_score - [N_D, 1] - [184, 1]
        Output: 
            pred_boxes - [N_D, 11] - [184, 11]
                Prediction result in kitti format
                [x1, y1, x2, y2, cx, cy, cz, w, h, l, alpha]
            mask - [N_D] - [184]
        '''
        # print(f"boxes = {boxes.shape}")
        # print(f"deltas = {deltas.shape}")
        # print(f"anchors_3d_mean_std = {anchors_3d_mean_std.shape}")
        # print(f"label_index = {label_index.shape}")
        # print(f"alpha_score = {alpha_score.shape}")
        # print(label_index)

        std = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1], dtype=torch.float32, device=boxes.device)
        
        # Anchor 2D boudning box
        widths  = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x   = boxes[..., 0] + 0.5 * widths
        ctr_y   = boxes[..., 1] + 0.5 * heights

        dx = deltas[..., 0] * std[0]
        dy = deltas[..., 1] * std[1]
        dw = deltas[..., 2] * std[2]
        dh = deltas[..., 3] * std[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        # Convert (x,y,w,h) to (x1, y1, x2, y2)
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        # TODO 
        one_hot_mask = torch.nn.functional.one_hot(label_index, anchors_3d_mean_std.shape[1]).bool()
        selected_mean_std = anchors_3d_mean_std[one_hot_mask] #[N]
        mask = selected_mean_std[:, 0, 0] > 0
        
        # Get cx, cy
        cdx = deltas[..., 4] * std[4]
        cdy = deltas[..., 5] * std[5]
        pred_cx1 = ctr_x + cdx * widths
        pred_cy1 = ctr_y + cdy * heights

        if cz_deltas != None:
            pred_z   = cz_deltas[...,0] * selected_mean_std[:, 0, 1] + selected_mean_std[:,0, 0]  #[N, 6]
            pred_sin = deltas[...,6]    * selected_mean_std[:, 1, 1] + selected_mean_std[:,1, 0] 
            pred_cos = deltas[...,7]    * selected_mean_std[:, 2, 1] + selected_mean_std[:,2, 0]
            pred_w   = deltas[...,8]    * selected_mean_std[:, 3, 1] + selected_mean_std[:,3, 0]
            pred_h   = deltas[...,9]    * selected_mean_std[:, 4, 1] + selected_mean_std[:,4, 0]
            pred_l   = deltas[...,10]   * selected_mean_std[:, 5, 1] + selected_mean_std[:,5, 0]
        else:
            pred_z   = deltas[...,6] * selected_mean_std[:, 0, 1] + selected_mean_std[:,0, 0]  #[N, 6]
            pred_sin = deltas[...,7] * selected_mean_std[:, 1, 1] + selected_mean_std[:,1, 0] 
            pred_cos = deltas[...,8] * selected_mean_std[:, 2, 1] + selected_mean_std[:,2, 0]
            pred_w   = deltas[...,9]  * selected_mean_std[:,3, 1] + selected_mean_std[:,3, 0]
            pred_h   = deltas[...,10] * selected_mean_std[:,4, 1] + selected_mean_std[:,4, 0]
            pred_l   = deltas[...,11] * selected_mean_std[:,5, 1] + selected_mean_std[:,5, 0]


        pred_alpha = torch.atan2(pred_sin, pred_cos) / 2.0
        
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2,
                                  pred_cx1, pred_cy1, pred_z,
                                  pred_w, pred_h, pred_l, pred_alpha], dim=1)
        
        pred_boxes[alpha_score[:, 0] < 0.5, -1] += np.pi

        return pred_boxes, mask
        
    def _sample(self, assignment_result, anchors, gt_bboxes):
        """
            I think this function currently do nothing. It suppose to balance out number of postive and negative samples
            Pseudo sampling
        """
        pos_inds = torch.nonzero(
                assignment_result['assigned_gt_inds'] > 0, as_tuple=False
            ).unsqueeze(-1).unique()
        neg_inds = torch.nonzero(
                assignment_result['assigned_gt_inds'] == 0, as_tuple=False
            ).unsqueeze(-1).unique()
        gt_flags = anchors.new_zeros(anchors.shape[0], dtype=torch.uint8) #

        pos_assigned_gt_inds = assignment_result['assigned_gt_inds'] - 1

        if gt_bboxes.numel() == 0:
            pos_gt_bboxes = gt_bboxes.new_zeros([0, 4])
        else:
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds[pos_inds], :]
        
        return_dict = dict(
            pos_inds = pos_inds,
            neg_inds = neg_inds,
            pos_bboxes = anchors[pos_inds],
            neg_bboxes = anchors[neg_inds],
            pos_gt_bboxes = pos_gt_bboxes,
            pos_assigned_gt_inds = pos_assigned_gt_inds[pos_inds],
        )
        return return_dict

    def _post_process(self, scores, bboxes, labels, P2s):
        
        N = len(scores)
        bbox2d = bboxes[:, 0:4]
        bbox3d = bboxes[:, 4:] #[cx, cy, z, w, h, l, alpha]

        bbox3d_state_3d = self.backprojector.forward(bbox3d, P2s[0]) #[x, y, z, w, h, l, alpha]
        for i in range(N):
            if bbox3d_state_3d[i, 2] > 3 and labels[i] == 0:
                bbox3d[i] = post_opt(
                    bbox2d[i], bbox3d_state_3d[i], P2s[0].cpu().numpy(),
                    bbox3d[i, 0].item(), bbox3d[i, 1].item()
                )
        bboxes = torch.cat([bbox2d, bbox3d], dim=-1)
        return scores, bboxes, labels

    def get_anchor(self, img_batch, P2):
        is_filtering = getattr(self.loss_cfg, 'filter_anchor', True)
        if not self.training:
            is_filtering = getattr(self.test_cfg, 'filter_anchor', is_filtering)
        # print(f"[detection_3d_head.py] is_filtering = {is_filtering}")
        
        anchors, useful_mask, anchor_mean_std = self.anchors(img_batch, P2, is_filtering=is_filtering)
        return_dict=dict(
            anchors=anchors, #[1, N, 4]
            mask=useful_mask, #[B, N]
            anchor_mean_std_3d = anchor_mean_std  #[N, C, K=6, 2]
        )
        return return_dict

    def _get_anchor_3d(self, anchors, anchor_mean_std_3d, assigned_labels):
        """
            anchors: [N_pos, 4] only positive anchors
            anchor_mean_std_3d: [N_pos, C, K=6, 2]
            assigned_labels: torch.Longtensor [N_pos, ]

            return:
                selected_mask = torch.bool [N_pos, ]
                selected_anchors_3d:  [N_selected, K, 2]
        """
        one_hot_mask = torch.nn.functional.one_hot(assigned_labels, self.num_classes).bool()
        selected_anchor_3d = anchor_mean_std_3d[one_hot_mask]

        selected_mask = selected_anchor_3d[:, 0, 0] > 0 #only z > 0, filter out anchors with good variance and mean
        selected_anchor_3d = selected_anchor_3d[selected_mask]

        return selected_mask, selected_anchor_3d

    def get_bboxes(self, preds, anchors, P2s, img_batch=None):
        '''
        Input: 
            * B: batch size
            * N_A: number of anchor
            cls_scores - [B, N_A, 2] - [1, 46080, 2]
                prediction of confidence score
            reg_preds - [B, N_A, 12] - [1, 46080, 12]
                prediction of boudning box
            anchors
            P2s
            img_batch - [B, 3, 288, 1280]

        Output: 
            * N_D: number of detection
            max_score - [N_D] - [5]
                confident score of detections
            bboxes - [N_D, 11] - [5, 11]
                bounding box of detections
            label - [N_D] - [5]
                class index, show category of detections 
        '''
        cls_scores = preds['cls_preds']
        reg_preds  = preds['reg_preds']
        cz_preds   = preds['cz_preds']
        noam_preds = preds['noam_preds']
        
        assert cls_scores.shape[0] == 1 # batch == 1
        
        # Parameters
        score_thr = getattr(self.test_cfg, 'score_thr', 0.5) # score_thr=0.75 in config.py
        # cls_agnostic: True -> directly NMS; False -> NMS with offsets different categories will not collide
        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # cls_agnostic = True 
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5) # nms_iou_thr=0.5
        is_post_opt = getattr(self.test_cfg, 'post_optimization', False) # post_optimization=True


        cls_scores  = cls_scores.sigmoid()
        cls_score   = cls_scores[0][..., 0:self.num_classes]
        alpha_score = cls_scores[0][..., self.num_classes:self.num_classes+1]
        reg_pred    = reg_preds[0]
        if self.is_seperate_cz: cz_preds = cz_preds[0]
        
        anchor = anchors['anchors'][0] #[N, 4]
        anchor_mean_std_3d = anchors['anchor_mean_std_3d'] #[N, K, 2]
        useful_mask = anchors['mask'][0] #[N, ]

        anchor             = anchor            [useful_mask]
        cls_score          = cls_score         [useful_mask]
        alpha_score        = alpha_score       [useful_mask]
        reg_pred           = reg_pred          [useful_mask]
        anchor_mean_std_3d = anchor_mean_std_3d[useful_mask] #[N, K, 2]
        if self.is_seperate_cz: cz_preds = cz_preds[useful_mask]

        # Find highest score in all classes
        max_score, label = cls_score.max(dim=-1) 
        high_score_mask = (max_score > score_thr)

        anchor             = anchor            [high_score_mask, :]
        anchor_mean_std_3d = anchor_mean_std_3d[high_score_mask, :]
        cls_score          = cls_score         [high_score_mask, :]
        alpha_score        = alpha_score       [high_score_mask, :]
        reg_pred           = reg_pred          [high_score_mask, :]
        max_score          = max_score         [high_score_mask]
        label              = label             [high_score_mask]
        if self.is_seperate_cz: cz_preds = cz_preds[high_score_mask]

        bboxes, mask = self._decode(anchor, reg_pred, anchor_mean_std_3d, label, alpha_score, cz_deltas=cz_preds)

        # Clip 2d bbox's boundary if exceed image.shape
        if img_batch is not None:
            bboxes = self.clipper(bboxes, img_batch)
        
        cls_score = cls_score[mask]
        max_score = max_score[mask]
        bboxes    = bboxes   [mask]

        # print(f"bboxes before nms = {bboxes.shape}") # [184, 11]
        if cls_agnostic:
            keep_inds = nms(bboxes[:, :4], max_score, nms_iou_thr)
        else:
            max_coordinate = bboxes.max()
            nms_bbox = bboxes[:, :4] + label.float().unsqueeze() * (max_coordinate)
            keep_inds = nms(nms_bbox, max_score, nms_iou_thr)

        bboxes      = bboxes   [keep_inds]
        max_score   = max_score[keep_inds]
        label       = label    [keep_inds]
        # print(f"bboxes after nms = {bboxes.shape}") # [1, 11]
        
        if is_post_opt:
            max_score, bboxes, label = self._post_process(max_score, bboxes, label, P2s)

        # print(f"Number of detection = {bboxes.shape[0]}")
        return max_score, bboxes, label, None

    def loss(self, preds, anchors, annotations, P2s):
        '''
        Input:
            preds is the preds result of the network:
                preds['cls_preds']  : [B, num_anchor, 2]
                preds['reg_preds']  : [B, num_anchor, 12]
                preds['cz_preds']   : [B, num_anchor, 1] or None 
                preds['noam_preds'] : [B, num_anchor, 8] or None
            
            anchors
                anchors['anchors'] : [1, num_anchor, 4]
                anchors['anchor_mean_std_3d'] : [1, num_anchor, ???]
            annotations: [B, max_lenght_label, 16], annotataions are ground trues # [8, 12, 16]
                [x1, y1, x2, y2, cls_index, cx, cy, z, w, h, l, alpha, x3d, y3d, z3d, rot_y]
        '''
        # print(f"loss annotations = {annotations.shape}") # [8, 12, 16]
        # cls_scores and reg_preds are predicted by netowrk 
        cls_scores = preds['cls_preds']
        reg_preds  = preds['reg_preds']
        cz_preds   = preds['cz_preds']
        noam_preds = preds['noam_preds']

        # annotataions are ground trues
        # print(f"anchors = {anchors['anchors'].shape}") # torch.Size([1, 46080, 4])
        # print(f"[detection_3d_head.py] cz_preds = {cz_preds.shape}") # [8, 46080, 1]
        batch_size = cls_scores.shape[0]
        anchor             = anchors['anchors'][0] #[N, 4]
        anchor_mean_std_3d = anchors['anchor_mean_std_3d']
        
        cls_loss = []
        reg_loss = []
        # iou_loss = []
        dep_loss = []
        nom_loss = []
        number_of_positives = []
        for j in range(batch_size):
            
            reg_pred    = reg_preds[j]
            cls_score   = cls_scores[j][..., 0:self.num_classes]
            alpha_score = cls_scores[j][..., self.num_classes:self.num_classes+1]
            if cz_preds != None: cz_pred       = cz_preds[j]
            if noam_preds != None: noam_pred = noam_preds[j]
            
            # selected by mask
            useful_mask = anchors['mask'][j] #[N]
            anchor_j             = anchor            [useful_mask]
            anchor_mean_std_3d_j = anchor_mean_std_3d[useful_mask]
            reg_pred             = reg_pred          [useful_mask]
            cls_score            = cls_score         [useful_mask]
            alpha_score          = alpha_score       [useful_mask]
            if cz_preds != None: cz_pred = cz_pred   [useful_mask]
            if noam_preds != None: noam_pred = noam_pred [useful_mask]

            # only select useful bbox_annotations
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]#[k]

            if len(bbox_annotation) == 0: # If this image doesn't contain any ground true, just skip it
                cls_loss.append(torch.tensor(0).cuda().float())
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
                # iou_loss.append(0.0) # This will make this very bad
                dep_loss.append(reg_preds.new_zeros(1)) # dep_loss.append(0.0)
                nom_loss.append(reg_preds.new_zeros(8))
                number_of_positives.append(0)
                continue
            
            # Filter 46080 anchors to 18974 anchors
            # print(f"Filter {anchor.shape[0]} anchors to {anchor_j.shape[0]} anchors")

            # print(f"anchor_j.shape = {anchor_j.shape}") # [7548, 4]
            # print(f"bbox_annotation.shape = {bbox_annotation.shape}") # [A, 12], where A is how many postive gts does this image have 

            # This is like retinanet's and YOLO's MaxIouAssignerm 
            assignement_result_dict = self._assign(anchor_j, bbox_annotation, **self.loss_cfg) # doesn't involve prediction
            
            # print(f"assignement_result_dict['num_gt'] = {assignement_result_dict['num_gt']}") # 4 
            # print(f"assignement_result_dict['assigned_gt_inds'] = {assignement_result_dict['assigned_gt_inds'].shape}") # [7548]
        
            # This is for checking GAC's anchor'a miss rate
            unique, counts = np.unique(assignement_result_dict['assigned_gt_inds'].cpu().numpy(), return_counts=True)
            anchor_assign = dict(zip(unique, counts)) # {-1: 100, 0: 3720, 1: 40}

            # print(f"assignement_result_dict['max_overlaps'] = {assignement_result_dict['max_overlaps'].shape}") # [7548]
            # print(f"assignement_result_dict['labels'] = {assignement_result_dict['labels'].shape}") # [7548]
            # print(assignement_result_dict['assigned_gt_inds'].unique()) # -1,  0,  1, 0 means negative, 1 meams positive, 

            # I think this sample function does nothing
            sampling_result_dict    = self._sample(assignement_result_dict, anchor_j, bbox_annotation) # doesn't involve prediction

            # print(f"sampling_result_dict['pos_bboxes'] = {sampling_result_dict['pos_bboxes'].shape}") # [40, 4], pos_bboxes are anchor that is assign to groundtrue 
            # print(f"sampling_result_dict['neg_bboxes'] = {sampling_result_dict['neg_bboxes'].shape}") # [3720, 4]
            # print(f"sampling_result_dict['pos_gt_bboxes'] = {sampling_result_dict['pos_gt_bboxes'].shape}") # [40, 12]
            # print(f"sampling_result_dict['pos_assigned_gt_inds'] = {sampling_result_dict['pos_assigned_gt_inds'].shape}") # [40]
            
            num_valid_anchors = anchor_j.shape[0]
            # print(f"num_valid_anchors = {num_valid_anchors}")
            labels = anchor_j.new_full((num_valid_anchors, self.num_classes),
                                    -1, # -1 not computed, binary for each class
                                    dtype=torch.float)

            pos_inds = sampling_result_dict['pos_inds']
            neg_inds = sampling_result_dict['neg_inds']
            #
            n_pos = pos_inds.shape[0]
            n_neg = neg_inds.shape[0]
            # print(f"n_pos, n_neg = {(n_pos, n_neg)}") # (86, 14078) n_pos, n_neg = (40, 3720)

            if len(pos_inds) > 0:
                pos_assigned_gt_label = bbox_annotation[sampling_result_dict['pos_assigned_gt_inds'], 4].long()
                
                selected_mask, selected_anchor_3d = self._get_anchor_3d(
                    sampling_result_dict['pos_bboxes'],
                    anchor_mean_std_3d_j[pos_inds],
                    pos_assigned_gt_label,
                )
                if len(selected_anchor_3d) > 0:
                    pos_inds = pos_inds[selected_mask]
                    pos_bboxes      = sampling_result_dict['pos_bboxes'][selected_mask]
                    pos_gt_bboxes   = sampling_result_dict['pos_gt_bboxes'][selected_mask] # pos_gt_bbox is the corresspondent target for that entry of postive bbox
                    pos_assigned_gt = sampling_result_dict['pos_assigned_gt_inds'][selected_mask]
                    # torch.set_printoptions(threshold=10_000)

                    # print(f"selected_mask = {selected_mask.shape}") # [627]
                    # print(f"pos_bboxes = {pos_bboxes.shape}") # [40, 4]
                    # print(f"pos_gt_bboxes = {pos_gt_bboxes.shape}") # [40, 12/24]
                    # print(f"selected_anchor_3d = {selected_anchor_3d.shape}") # [40, 6, 2]
                    
                    # import pickle
                    # with open("GAC_head_anchor_2D.pkl", 'wb') as f:
                    #     pickle.dump(pos_bboxes.detach().cpu().numpy(), f)
                    # with open("GAC_head_anchor_3D.pkl", 'wb') as f:
                    #     pickle.dump(selected_anchor_3d.detach().cpu().numpy(), f)
                    # print(f"Output anchor's information to GAC_head_anchor_2D.pkl and GAC_head_anchor_3D.pkl")

                    pos_bbox_targets, targets_alpha_cls = self._encode(
                        pos_bboxes, pos_gt_bboxes, selected_anchor_3d
                    ) #[N, 12], [N, 1]
                    label_index = pos_assigned_gt_label[selected_mask]
                    labels[pos_inds, :] = 0
                    labels[pos_inds, label_index] = 1

                    pos_anchor = anchor[pos_inds]
                    pos_alpha_score = alpha_score[pos_inds]

                    # print(f"self.decode_before_loss = {self.decode_before_loss}") # False 
                    # if self.decode_before_loss:
                    #     pos_prediction_decoded, mask = self._decode(pos_anchor, reg_pred[pos_inds],  anchor_mean_std_3d_j[pos_inds], label_index, pos_alpha_score)
                    #     pos_target_decoded, _     = self._decode(pos_anchor, pos_bbox_targets,  anchor_mean_std_3d_j[pos_inds], label_index, pos_alpha_score)
                    #     reg_loss_j = self.loss_bbox(pos_prediction_decoded[mask], pos_target_decoded[mask])
                    #     alpha_loss_j = self.alpha_loss(pos_alpha_score[mask], targets_alpha_cls[mask])
                    #     loss_j = torch.cat([reg_loss_j, alpha_loss_j], dim=1) * self.regression_weight #[N, 12]
                    #     reg_loss.append(loss_j.mean(dim=0)) #[13]
                    #     number_of_positives.append(bbox_annotation.shape[0])
                    # else:

                    ######################
                    ### Get Depth Loss ###
                    ######################
                    if self.is_seperate_cz:
                        dep_target = pos_bbox_targets[:, 6:6+1] # [697, 1] 
                        if self.cz_pred_mode == "look_ground" or self.cz_pred_mode == "fc":
                            dep_loss_j = self.loss_bbox(dep_target, cz_pred[pos_inds]) # [141, 1]
                        
                        elif self.cz_pred_mode == "oridinal_loss":
                            raise NotImplementedError # TODO
                        
                        # print(f"dep_loss_j.requires_grad = {dep_loss_j.requires_grad}")
                        dep_loss.append( dep_loss_j.mean(dim=0) * 3) # TODO, this 3 ratio is derived from regersssion_weight
                        # 
                        pos_bbox_targets = torch.cat((pos_bbox_targets[:, :6], pos_bbox_targets[:, 6+1:]), dim=1) # [694, 11]
                    
                    ###########################
                    ### Get Regression Loss ###
                    ###########################
                    # print(f"pos_bbox_targets = {pos_bbox_targets.shape}") # [628, 12]
                    # print(f"reg_pred[pos_inds] = {reg_pred[pos_inds].shape}") # [628, 20]
                    if self.is_noam_loss and (not self.is_seperate_noam):
                        reg_loss_j   = self.loss_bbox(pos_bbox_targets, reg_pred[pos_inds][:, :12])
                        alpha_loss_j = self.alpha_loss(pos_alpha_score, targets_alpha_cls)
                        noam_loss_j  = self.loss_bbox(pos_gt_bboxes[:, 16:], reg_pred[pos_inds][:, 12:])
                        loss_j = torch.cat([reg_loss_j, alpha_loss_j, noam_loss_j], dim=1) * self.regression_weight #[N, 13]
                    else:
                        reg_loss_j   = self.loss_bbox(pos_bbox_targets, reg_pred[pos_inds]) # This is the first time, loss() used prediction result
                        alpha_loss_j = self.alpha_loss(pos_alpha_score, targets_alpha_cls)
                        loss_j = torch.cat([reg_loss_j, alpha_loss_j], dim=1) * self.regression_weight #[N, 13]
                    
                    reg_loss.append(loss_j.mean(dim=0)) #[13] / [25]
                    number_of_positives.append(bbox_annotation.shape[0])

                    ####################
                    ### Get IOU Loss ###
                    ####################
                    # pos_prediction_decoded, mask = self._decode(pos_anchor, reg_pred[pos_inds],  anchor_mean_std_3d_j[pos_inds], label_index, pos_alpha_score)
                    # pos_target_decoded, _        = self._decode(pos_anchor, pos_bbox_targets,    anchor_mean_std_3d_j[pos_inds], label_index, pos_alpha_score)
                    # loss_iou_j = self.loss_iou(pos_prediction_decoded[mask, :4], pos_target_decoded[mask, :4])
                    # iou_loss.append(loss_iou_j.mean().item())
                    
                    #####################
                    ### Get NOAM Loss ###
                    #####################
                    if self.is_noam_loss and self.is_seperate_noam:
                        nom_loss_j = self.loss_bbox(pos_gt_bboxes[:, 16:], noam_pred[pos_inds]) # [141, 1]
                        nom_loss.append( nom_loss_j.mean(dim=0) * 3)
            else:
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
                number_of_positives.append(bbox_annotation.shape[0])
                # iou_loss.append(0.0)
                if cz_preds != None:   dep_loss.append(reg_preds.new_zeros(1))
                if noam_preds != None: nom_loss.append(reg_preds.new_zeros(8))

            if len(neg_inds) > 0:
                labels[neg_inds, :] = 0
            
            # Get classification loss
            cls_loss.append(self.loss_cls(cls_score, labels).sum() / (len(pos_inds) + len(neg_inds)))
        
        cls_loss = torch.stack(cls_loss).mean(dim=0, keepdim=True)
        reg_loss = torch.stack(reg_loss, dim=0) #[B, 12]
        # iou_loss = reg_loss.new(iou_loss)
        
        # print(f"Befroe dep_loss.requires_grad = {dep_loss[0].requires_grad}")
        if cz_preds != None:   dep_loss = torch.stack(dep_loss, dim=0)
        if noam_preds != None: nom_loss = torch.stack(nom_loss, dim=0)
        
        # Weight regression loss by number of ground true in each images
        weights = reg_pred.new(number_of_positives).unsqueeze(1) #[B, 1]
        weighted_regression_losses = torch.sum(weights * reg_loss / (torch.sum(weights) + 1e-6), dim=0)
        reg_loss = weighted_regression_losses.mean(dim=0, keepdim=True)
        
        # Weight IoU loss by number of ground true in each images
        # weighted_iou_losses = torch.sum(weights * iou_loss / (torch.sum(weights) + 1e-6), dim=0)
        # iou_loss = weighted_iou_losses.mean(dim=0, keepdim=True)

        # Weight Depth loss by number of ground true in each images
        if cz_preds != None:
            weighted_dep_losses = torch.sum(weights * dep_loss / (torch.sum(weights) + 1e-6), dim=0)
            dep_loss = weighted_dep_losses.mean(dim=0, keepdim=True)
            # print(f"dep_loss.requires_grad = {dep_loss.requires_grad}")
        
        if noam_preds != None:
            weighted_nom_losses = torch.sum(weights * nom_loss / (torch.sum(weights) + 1e-6), dim=0)
            nom_loss = weighted_nom_losses.mean(dim=0, keepdim=True)
        
        # TODO, re-implement iou_loss someday, also 2d bbox regression can be eliminated if we use iou loss
        log_dict = {}
        log_dict['1/cls_loss']   = cls_loss
        log_dict['1/reg_loss']   = reg_loss
        log_dict['1/total_loss'] = cls_loss + reg_loss
        if self.is_seperate_cz:
            log_dict['1/dep_loss']    = dep_loss
            log_dict['1/total_loss'] += dep_loss
        if self.is_noam_loss and self.is_seperate_noam:
            log_dict['1/nom_loss']    = nom_loss
            log_dict['1/total_loss'] += nom_loss
        log_dict['2/dx']     = weighted_regression_losses[0]
        log_dict['2/dy']     = weighted_regression_losses[1]
        log_dict['2/dw']     = weighted_regression_losses[2]
        log_dict['2/dh']     = weighted_regression_losses[3]
        log_dict['2/cdx']    = weighted_regression_losses[4]
        log_dict['2/cdy']    = weighted_regression_losses[5]
        if self.is_seperate_cz:
            log_dict['2/cdz']    = dep_loss.detach().cpu().numpy()[0]
            log_dict['4/d_sin']  = weighted_regression_losses[6]
            log_dict['4/d_cos']  = weighted_regression_losses[7]
            log_dict['4/dw']     = weighted_regression_losses[8]
            log_dict['4/dh']     = weighted_regression_losses[9]
            log_dict['4/dl']     = weighted_regression_losses[10]
            log_dict['4/d_hdg']  = weighted_regression_losses[11]
        else:
            log_dict['2/cdz']    = weighted_regression_losses[6]
            log_dict['4/d_sin']  = weighted_regression_losses[7]
            log_dict['4/d_cos']  = weighted_regression_losses[8]
            log_dict['4/dw']     = weighted_regression_losses[9]
            log_dict['4/dh']     = weighted_regression_losses[10]
            log_dict['4/dl']     = weighted_regression_losses[11]
            log_dict['4/d_hdg']  = weighted_regression_losses[12]
        
        if self.is_noam_loss and (not self.is_seperate_noam):
            log_dict['5/farer_x']  = weighted_regression_losses[12]
            log_dict['5/farer_y']  = weighted_regression_losses[13]
            log_dict['5/close_x']  = weighted_regression_losses[14]
            log_dict['5/close_y']  = weighted_regression_losses[15]
            log_dict['5/right_x']  = weighted_regression_losses[16]
            log_dict['5/right_y']  = weighted_regression_losses[17]
            log_dict['5/lefft_x']  = weighted_regression_losses[18]
            log_dict['5/lefft_y']  = weighted_regression_losses[19]
        
        log_dict['3/n_positive_anchor'] = torch.tensor([n_pos])
        log_dict['3/n_negative_anchor'] = torch.tensor([n_neg])
        # log_dict['3/n_ignored_anchor']  = torch.tensor([n_ign])
        log_dict['3/n_positive_predict'] = torch.numel(cls_score[ cls_score >  0 ])
        log_dict['3/n_negative_predict'] = torch.numel(cls_score[ cls_score <= 0 ])
        log_dict['3/mean_positive_predict'] = cls_score[ cls_score >  0 ].mean()
        log_dict['3/mean_negative_predict'] = cls_score[ cls_score <= 0 ].mean()

        return log_dict

class StereoHead(AnchorBasedDetection3DHead):
    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):

        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(cls_feature_size, num_anchors*(num_cls_output), kernel_size=3, padding=1),
            AnchorFlatten(num_cls_output)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction = nn.Sequential(
            ConvBnReLU(num_features_in, reg_feature_size, (3, 3)),
            BasicBlock(reg_feature_size, reg_feature_size),
            nn.ReLU(),
            nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten(num_reg_output)
        )

        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

class GroundAwareHead(AnchorBasedDetection3DHead):
    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):
        
        ####################################
        ### Attention Module Declaration ###
        ####################################
        if self.use_spatial_attention or self.use_channel_attention:
            self.cbam_layer = CBAM(num_features_in, 
                                    reduction_ratio = 16, 
                                    pool_types = ['avg', 'max'], 
                                    use_spatial = self.use_spatial_attention, 
                                    use_channel = self.use_channel_attention)
            print(self.cbam_layer)
        
        if self.use_bam:
            self.bam_layer = BAM(num_features_in)
            print(self.bam_layer)
        
        if self.use_coordinate_attetion:
            self.coord_atten_layer = CoordAtt(num_features_in, num_features_in)
            print(self.coord_atten_layer)
        
        ###################################
        ### Dilation Module Declaration ###
        ###################################
        if self.num_dcnv2 != 0:
            self.dcn_layers = nn.ModuleList([
                DeformableConv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
                for _ in range(self.num_dcnv2)
            ])

        if self.num_pac_layer != 0:
            if self.is_pac_bn_relu:
                self.pac_layers = []
                for _ in range(self.num_pac_layer):
                    self.pac_layers.append( nn.Sequential(
                        PerspectiveConv2d(num_features_in,
                                            num_features_in,
                                            self.pac_mode,
                                            offset_2d = self.offset_2d,
                                            offset_3d = self.offset_3d,
                                            input_shape = (18, 80),
                                            pad_mode=self.pad_mode,
                                            adpative_P2 = self.adpative_P2,),
                        nn.BatchNorm2d(num_features_in),
                        nn.ReLU(),) )
                self.pac_layers = [n.to("cuda") for n in self.pac_layers] # TODO, tmp
            else:
                self.pac_layers = nn.ModuleList([
                    PerspectiveConv2d(num_features_in,
                                    num_features_in,
                                    self.pac_mode,
                                    offset_2d = self.offset_2d,
                                    offset_3d = self.offset_3d,
                                    input_shape = (18, 80),
                                    pad_mode=self.pad_mode,
                                    adpative_P2 = self.adpative_P2,)
                    for _ in range(self.num_pac_layer) ]) 
                
            print(f"self.pac_layers = {self.pac_layers}")

        #########################################
        ### Classification Branch Declaration ###
        #########################################
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

        print(f"GroundAwareHead self.exp = {self.exp}")
        assert self.exp != "no_look_ground", f"self.exp == no_look_ground, this setting is not supported anymore, because it's worse than baseline"

        #####################################
        ### Regression Branch Declaration ###
        #####################################
        self.reg_feature_extraction = nn.Sequential(
            LookGround(num_features_in, self.exp),
            nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(),
            nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten((num_reg_output))
        )
        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

        ################################
        ### Depth Branch Declaration ###
        ################################
        if self.is_seperate_cz:
            if self.cz_pred_mode == "fc":
                self.cz_feature_extraction = nn.Sequential( # Reduce dimension from 1024 to 256
                    nn.Conv2d(num_features_in, 256, kernel_size = 1),
                )
                self.cz_fc_layer = nn.Sequential(
                    nn.Linear(18*80*256, 1440),
                    nn.ReLU(),
                    nn.Linear(1440 , 1440),
                )
            elif self.cz_pred_mode == "look_ground":
                self.cz_feature_extraction = nn.Sequential(
                    LookGround(num_features_in, self.exp),
                    nn.Conv2d(num_features_in, self.cz_reg_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.cz_reg_dim),
                    nn.ReLU(),
                    nn.Conv2d(self.cz_reg_dim, self.cz_reg_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.cz_reg_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.cz_reg_dim, num_anchors*1, kernel_size=3, padding=1),
                    AnchorFlatten((1)),
                )
                self.cz_feature_extraction[-2].weight.data.fill_(0)
                self.cz_feature_extraction[-2].bias.data.fill_(0)
            
            elif self.cz_pred_mode == "oridinal_loss": # TODO
                raise NotImplementedError
        
        ###################################
        ### Noam Regression Declaration ###
        ###################################
        if self.is_noam_loss and self.is_seperate_noam:
            self.noam_feature_extraction = nn.Sequential(
                nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(),
                nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(reg_feature_size, num_anchors*8, kernel_size=3, padding=1),
                AnchorFlatten((8))
            )
            self.noam_feature_extraction[-2].weight.data.fill_(0)
            self.noam_feature_extraction[-2].bias.data.fill_(0)
        
    def forward(self, inputs):
        ########################
        ### Attention Module ###
        ########################
        if self.use_spatial_attention or self.use_channel_attention:
            inputs['features'] = self.cbam_layer(inputs['features']) # [1024, 18, 80]
        
        if self.use_bam:
            inputs['features'] = self.bam_layer(inputs['features'])
        
        if self.use_coordinate_attetion:
            inputs['features'] = self.coord_atten_layer(inputs['features'])
        
        ########################
        ### Dilation  Module ###
        ########################
        if self.num_dcnv2 != 0:
            for i in range(self.num_dcnv2):
                inputs['features'] = self.dcn_layers[i](inputs['features'])
        
        if self.num_pac_layer != 0:
            for i in range(self.num_pac_layer):
                inputs['features'] = self.pac_layers[i](inputs)
        
        cls_preds = self.cls_feature_extraction(inputs['features'])
        if self.exp == "no_look_ground":
            reg_preds = self.reg_feature_extraction(inputs['features'])
        else:
            reg_preds = self.reg_feature_extraction(inputs)
        
        ################################
        ### Depth Prediction  Module ###
        ################################
        if self.is_seperate_cz:
            if self.cz_pred_mode == "fc":
                cz_preds = self.cz_feature_extraction(inputs['features'])
                cz_preds = cz_preds.permute(0, 2, 3, 1)
                cz_preds = cz_preds.contiguous().view(cz_preds.shape[0], -1) # torch.Size([8, 368640])
                cz_preds = self.cz_fc_layer(cz_preds) # torch.Size([8, 1440])
                cz_preds = cz_preds.view(cz_preds.shape[0], -1, 18, 80) # torch.Size([8, 1, 18, 80])
                cz_preds = cz_preds.repeat(1, 32, 1, 1) # torch.Size([8, 32, 18, 80])
            
            elif self.cz_pred_mode == "look_ground":
                cz_preds = self.cz_feature_extraction(inputs)
            
            elif self.cz_pred_mode == "oridinal_loss":
                raise NotImplementedError
        else:
            cz_preds = None
        
        # if self.is_seperate_cz:
        #     # print(f"cz_preds = {cz_preds.shape}") # torch.Size([8, 32, 18, 80])
        #     # print(f"reg_preds = {reg_preds.shape}") # torch.Size([8, 352, 18, 80])
            
        #     # Adjust Dimension to [B, num_anchor, num_regression_value, H, W]
        #     reg_preds = reg_preds.view(reg_preds.shape[0], self.layer_cfg.num_anchors, -1, reg_preds.shape[2], reg_preds.shape[3])
        #     if self.is_seperate_cz:
        #         cz_preds  = torch.unsqueeze(cz_preds, 2)
            
        #     # print(f"reg_preds_new = {reg_preds.shape}") # torch.Size([8, 32, 11, 18, 80])
        #     # print(f"cz_preds_new = {cz_preds.shape}") #  torch.Size([8, 32, 1, 18, 80])
        #     # [dx, dy, dw, dh, cdx, cdy, cdz, cd_sin, cd_cos, w3d, h3d, l3d]
        #     #  0   1   2   3   4    5    6    7       8       9    10   11
            
        #     # Concat seperate value together
        #     if self.is_seperate_cz:
        #         reg_preds = torch.cat([reg_preds[:, :, :6 , :, :], cz_preds , reg_preds[:, :, 6: , :, :]], dim=2)
        #     # print(f"all_reg_preds = {all_reg_preds.shape}") # torch.Size([8, 32, 12, 18, 80])
        #     reg_preds = reg_preds.view(reg_preds.shape[0], -1, reg_preds.shape[3], reg_preds.shape[4])
        #     # print(f"reg_preds = {reg_preds.shape}") # torch.Size([8, 384, 18, 80])
        #     # reg_preds_new = torch.tensor_split(reg_preds, 32, dim=1)
        
        # Anchor Flattend
        # reg_preds = reg_preds.permute(0, 2, 3, 1)
        # reg_preds = reg_preds.contiguous().view(reg_preds.shape[0], -1, 12)
        
        # print(f"cls_preds = {cls_preds.shape}") # [B, 46080, 2]
        # print(f"reg_preds = {reg_preds.shape}") # [B, 46080, 11]
        # print(f"cz_preds  = {cz_preds.shape}")  # [B, 46080, 1]
        
        ###################
        ### NOAM Moduel ###
        ###################
        if self.is_noam_loss and self.is_seperate_noam:
            noam_preds = self.noam_feature_extraction(inputs['features'])
        else:
            noam_preds = None
        
        return {"cls_preds" : cls_preds,
                "reg_preds" : reg_preds,
                "cz_preds"  : cz_preds,
                "noam_preds": noam_preds}
