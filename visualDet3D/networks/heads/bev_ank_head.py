import numpy as np
import pickle
from easydict import EasyDict
from math import pi

import torch
import torch.nn as nn
from torchvision.ops import nms

from visualDet3D.networks.heads.losses import SigmoidFocalLoss, ModifiedSmoothL1Loss
from visualDet3D.networks.utils.utils import calc_iou, BackProjection
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.networks.utils.utils import ClipBoxes
from visualDet3D.networks.lib.look_ground import LookGround

from visualDet3D.utils.util_kitti import gac_original_anchor_parser, STD_HEIGT, STD_WIDTH, STD_LENTH
import os

DEVICE = torch.device("cuda:0") # TODO

P2_tf = np.array([[ 7.55646755e+02, 0.00000000e+00, 6.38374831e+02, 4.69778060e+01],
                  [ 0.00000000e+00, 7.55646755e+02, 7.62980073e+01, -6.09610124e-02],
                  [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.74588400e-03]])

class BeVAnchorFlatten(nn.Module):
    """
        Module for anchor-based network outputs,
        Init args:
            num_output: number of output channel for each anchor.

        Forward args:
            x: torch.tensor of shape [B, num_anchors * output_channel, H, W]

        Forward return:
            x : torch.tensor of shape [B, num_anchors * H * W, output_channel]
    """
    def __init__(self, anchor_mask):
        super(BeVAnchorFlatten, self).__init__()
        # self.anchor_distribution = anchor_distribution
        self.anchor_mask = anchor_mask
        # self.n_anchor   = int(self.anchor_distribution.sum())
        # self.max_anchor = int(self.anchor_distribution.max()) # 144
        # print(f"Number of anchor = {self.n_anchor}")
        
        # # Build anchor_bool_mapm, TODO, is this the culprit?!?!?!?!??!!
        # self.anchor_bool_map = torch.zeros(18*80*self.max_anchor , dtype=torch.bool)
        # for i in range(self.anchor_distribution.shape[0]): # 18
        #     for j in range(self.anchor_distribution.shape[1]): # 80
        #         start_idx_reg = (i*80 + j)*self.max_anchor
                
        #         n_anchor_pixel = int(self.anchor_distribution[i, j])
        #         assert n_anchor_pixel <= self.max_anchor, "GG!!! n_anchor_pixel exceed max number of anchor"
                
        #         self.anchor_bool_map[start_idx_reg : start_idx_reg + n_anchor_pixel] = True
        
    def forward(self, x):
        '''
        x = [1, 1728, 18, 80]
        '''
        x = x.permute(0, 2, 3, 1) # [B, 18, 80, 1728]
        x = x.contiguous().view(x.shape[0], self.anchor_mask.shape[0], -1) # [B, 207360, 12] or [B, 207360, 2]
        # Remember to make sure the anchor is row-major
        
        # Filter invalid anchor
        x = x[:, self.anchor_mask, :] # [B, 14260, 12] or [B, 14260, 2]
        return x

class BevAnk3DHead(nn.Module):
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
                       data_cfg:EasyDict=EasyDict(),):
        super(BevAnk3DHead, self).__init__()
        self.num_classes = num_classes
        self.num_regression_loss_terms=num_regression_loss_terms
        self.decode_before_loss = getattr(loss_cfg, 'decode_before_loss', False)
        self.loss_cfg = loss_cfg
        self.test_cfg  = test_cfg
        self.build_loss(**loss_cfg)
        self.backprojector = BackProjection()
        self.clipper = ClipBoxes()
        self.anchor_assignment_method = loss_cfg.anchor_assignment
        self.anchor_generation_method = loss_cfg.anchor_generation
        self.exp = exp
        print(f"anchor_assignment_method = {self.anchor_assignment_method}")
        print(f"anchor_generation_method = {self.anchor_generation_method}")

        if self.anchor_generation_method == 'gac_anchor':
            anchor_2D, anchor_3D, self.anchor_mask, anchor_objects = gac_original_anchor_parser(f"{data_cfg.anchor_mean_std_path}_anchor.pkl", is_transform = True)
            anchors_tensor = []
            for a in anchor_objects:
                anchors_tensor.append([a.xmin, a.ymin, a.xmax, a.ymax, 0, a.cx, a.cy, a.cz, a.w, a.h , a.l, a.alpha, a.x3d, a.y3d, a.z3d, a.rot_y])
            self.anchors = torch.tensor(anchors_tensor).to(DEVICE)
            self.n_anchor = self.anchors.shape[0]
            print(f"anchors_tensor = {self.anchors.shape}") # [19336, 16]

            # Get anchor distribution 
            # self.anchor_distribution = torch.full((18, 80), 32)
            # self.anchor_mask = GAC_head_anchor_mask

        else:
            # Load anchor's bbox 
            anchor_path = os.path.join("/home/lab530/KenYu/ml_toolkit/anchor_generation/pkl/", 
                                       f"{self.anchor_generation_method}.pkl")
            with open(anchor_path, 'rb') as f:
                self.anchors = pickle.load(f).to(DEVICE)
            print(f"Load anchors.pkl from {anchor_path}")
            self.n_anchor = self.anchors.shape[0] # Number of anchor
            print(f"self.anchors = {self.anchors.shape}")

            # Load anchor_mask.pkl
            anchor_map_path = os.path.join("/home/lab530/KenYu/ml_toolkit/anchor_generation/pkl/", 
                                           f"{self.anchor_generation_method}_map.pkl")
            with open(anchor_map_path, 'rb') as f:
                self.anchor_mask = pickle.load(f)
            print(f"Load anchor_mask.pkl from {anchor_map_path}")
            
        ####################################################################
        # Make sure they have the same number of anchors
        assert self.n_anchor == int(torch.count_nonzero(torch.tensor(self.anchor_mask))), f"Anchor bboxes and anchor distribution have different number of anchor! {self.n_anchor} != {int(torch.count_nonzero(torch.tensor(self.anchor_mask)))}"
        # 
        self.init_layers(**layer_cfg)

    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):
        '''
            num_features_in=1024,
            num_cls_output=len(cfg.obj_types)+1,
            num_reg_output=12,
            cls_feature_size=512,
            reg_feature_size=1024,
        '''
        # n_max_anchor = int(self.anchor_mask.max()) # 144
        n_max_anchor = int( self.anchor_mask.shape[0] / 1440 )
        print(f"Maximum anchor concentration = {n_max_anchor}")

        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            
            # TODO 'groups' argument might be abel to help ???????
            nn.Conv2d(cls_feature_size, n_max_anchor*num_cls_output, kernel_size=3, padding=1),
            BeVAnchorFlatten(self.anchor_mask)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction = nn.Sequential(
            LookGround(num_features_in, self.exp),
            nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(),
            nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),

            nn.Conv2d(reg_feature_size, n_max_anchor*num_reg_output, kernel_size=3, padding=1), # 1728
            BeVAnchorFlatten(self.anchor_mask)
        )
        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

    def forward(self, inputs):
        cls_preds = self.cls_feature_extraction(inputs['features'])
        reg_preds = self.reg_feature_extraction(inputs) # inputs['features']
        return cls_preds, reg_preds

    def build_loss(self, focal_loss_gamma=0.0, balance_weight=[0], L1_regression_alpha=9, **kwargs):
        self.focal_loss_gamma = focal_loss_gamma
        self.register_buffer("balance_weights", torch.tensor(balance_weight, dtype=torch.float32))
        self.cls_loss = SigmoidFocalLoss(gamma=focal_loss_gamma, balance_weights=self.balance_weights)
        self.reg_loss = ModifiedSmoothL1Loss(L1_regression_alpha)

        regression_weight = kwargs.get("regression_weight", [1 for _ in range(self.num_regression_loss_terms)]) #default 12 only use in 3D
        self.register_buffer("regression_weight", torch.tensor(regression_weight, dtype=torch.float))

        self.hdg_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _encode(self, aks, gts):
        assert aks.shape[0] == gts.shape[0]
        '''
        This function calcuate difference between anchor and ground true -> encode into format that network should be predicting
        '''
        # gts = [x1, y1, x2, y2, cls_index, cx, cy , cz, w, h, l , alpha]
        #        0  1   2   3   4          5   6    7   8  9  10, 11

        aks = aks.float()
        gts = gts.float()

        # Convert (x1, y1, x2, y2) to (x_c, y_c, w, h)
        ax = (aks[..., 0] + aks[..., 2]) * 0.5
        ay = (aks[..., 1] + aks[..., 3]) * 0.5
        aw =  aks[..., 2] - aks[..., 0]
        ah =  aks[..., 3] - aks[..., 1]
        gx = (gts[..., 0] + gts[..., 2]) * 0.5
        gy = (gts[..., 1] + gts[..., 3]) * 0.5
        gw =  gts[..., 2] - gts[..., 0]
        gh =  gts[..., 3] - gts[..., 1]

        # diff of 2D bbox's center and geometry 
        targets_dx = (gx - ax) / aw
        targets_dy = (gy - ay) / ah
        targets_dw = torch.log(gw / aw)
        targets_dh = torch.log(gh / ah)

        # 3D bbox center on image plane
        # TODO, GAC use 2d bbox center to find cx cy, I believe it's because that anchor is defined on that pixel location
        # However, in my implementation, I decided to use (cx, cy) to define anchor's feature's lcoation on image
        # multiply by aw to avoid value become too large
        targets_cdx = (gts[:, 5] - aks[:, 5]) / aw
        targets_cdy = (gts[:, 6] - aks[:, 6]) / ah

        # 3D bbox center's depth
        if self.loss_cfg.is_norm_cz:
            targets_cdz = (gts[:, 7] - aks[:, 7]) * 1292.1559425*aks[:, 7]**(-2) / ah
        else:
            targets_cdz = gts[:, 7] - aks[:, 7]
        
        # 3D bbox orientation, TODO, is 2x really nesseary? 
        targets_cd_sin = torch.sin(gts[:, 11] * 2) - torch.sin(aks[:, 11] * 2)
        targets_cd_cos = torch.cos(gts[:, 11] * 2) - torch.cos(aks[:, 11] * 2)
        
        # 3D bbox geometry
        targets_w3d = (gts[:, 8]  - aks[:, 8] ) / STD_WIDTH
        targets_h3d = (gts[:, 9]  - aks[:, 9] ) / STD_HEIGT
        targets_l3d = (gts[:, 10] - aks[:, 10]) / STD_LENTH
        
        # Get target values for network to predict
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh,
                               targets_cdx, targets_cdy, targets_cdz,
                               targets_cd_sin, targets_cd_cos,
                               targets_w3d, targets_h3d, targets_l3d), dim=1)
        
        targets_hdg = (torch.cos(gts[:, 11:12]) > 0).float() # This decide the object is heading front or back
        
        return targets, targets_hdg

    def _decode(self, aks, pds, hdg_prediction):
        '''
        Input: 
            * N_D: number of detection 
            aks - [N_D, 12] - [184, 12]
                [x1, y1, x2, y2, cls_index, cx, cy , cz, w, h, l , alpha]
                anchor of these detection corresspond to 
            pds - [N_D, 12] - [184, 12]
                [dx, dy, dw, dh, cdx, cdy, cdz, cd_sin, cd_cos, w3d, h3d, l3d]
                prediction result
            hdg_prediction - [N_D, 1] - [184, 1]
        Output: 
            out_boxes - [N_D, 11] - [184, 11]
                Prediction result in kitti format
                [x1, y1, x2, y2, cx, cy, cz, w, h, l, alpha]
        '''

        # std = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1], dtype=torch.float32, device=boxes.device)
        
        # Convert Anchor's 2D bbox from (x1,y1,x2,y2) to (x,y,w,h)
        ak_w = aks[..., 2] - aks[..., 0]
        ak_h = aks[..., 3] - aks[..., 1]
        ak_x = aks[..., 0] + 0.5 * ak_w
        ak_y = aks[..., 1] + 0.5 * ak_h

        out_x = ak_x + pds[..., 0] * ak_w
        out_y = ak_y + pds[..., 1] * ak_h
        out_w = torch.exp(pds[..., 2]) * ak_w
        out_h = torch.exp(pds[..., 3]) * ak_h

        # Convert (x,y,w,h) to (x1, y1, x2, y2)
        out_x1 = out_x - 0.5 * out_w
        out_y1 = out_y - 0.5 * out_h
        out_x2 = out_x + 0.5 * out_w
        out_y2 = out_y + 0.5 * out_h
        
        # Get cx, cy, 
        out_cx = aks[:, 5] + pds[..., 4] * ak_w
        out_cy = aks[:, 6] + pds[..., 5] * ak_h

        # Get z
        if self.loss_cfg.is_norm_cz:
            out_z = pds[..., 6] / (1292.1559425*aks[:, 7]**(-2) / ak_h) + aks[:, 7]
        else:
            out_z = aks[:, 7] + pds[..., 6]
        
        # Get alpha
        out_sin = pds[...,7] + torch.sin(aks[:, 11] * 2)
        out_cos = pds[...,8] + torch.cos(aks[:, 11] * 2)
        out_alpha = torch.atan2(out_sin, out_cos) / 2.0

        # Get dimension of 3d bbox
        out_w = pds[...,9]  * STD_WIDTH + aks[:, 8]
        out_h = pds[...,10] * STD_HEIGT + aks[:, 9]
        out_l = pds[...,11] * STD_LENTH + aks[:, 10]

        out_boxes = torch.stack([out_x1, out_y1, out_x2, out_y2,
                                  out_cx, out_cy, out_z,
                                  out_w, out_h, out_l, out_alpha], dim=1)
        
        # Use predicted alpha heading to revise ????? TODO
        out_boxes[hdg_prediction[:, 0] < 0.5, -1] += np.pi

        return out_boxes

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
        '''
        This is a dummy function rightnow because i don't want to generate anchor for every iteretion
        '''
        return None

    def get_bboxes(self, cls_preds, reg_preds, anchors_dummy, P2s, img_batch=None):
        '''
        This function is call by test_forward, test version of loss()
        
        Input: 
            * B: batch size
            * N_A: number of anchor
            cls_preds - [B, N_A, 2] - [1, 46080, 2]
                prediction of confidence score
            reg_preds - [B, N_A, 12] - [1, 46080, 12]
                prediction of boudning box
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
        # Batch size must be 1
        assert cls_preds.shape[0] == 1 and reg_preds.shape[0] == 1
        
        # Parameters
        score_thr = getattr(self.test_cfg, 'score_thr', 0.5) # score_thr=0.75 in config.py
        # cls_agnostic: True -> directly NMS; False -> NMS with offsets different categories will not collide
        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # cls_agnostic = True 
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5) # nms_iou_thr=0.5
        is_post_opt = getattr(self.test_cfg, 'post_optimization', False) # post_optimization=True

        cls_preds = cls_preds.sigmoid()
        # print(f"cls_preds = {cls_preds.shape}") # [1, 9332, 2]
        cls_pred = cls_preds[0][..., 0:self.num_classes]
        hdg_pred = cls_preds[0][..., self.num_classes:self.num_classes+1]
        reg_pred = reg_preds[0]
        
        # Find highest score in all classes
        max_score, label = cls_pred.max(dim=-1) # [14284], [14284]
        # print(max_score.max())

        # Confident score much high than score_thr to be a postive detection
        high_score_mask = (max_score > score_thr)

        anchor_det= self.anchors[high_score_mask, :]
        cls_pred  = cls_pred[high_score_mask, :]
        hdg_pred  = hdg_pred[high_score_mask, :]
        reg_pred  = reg_pred[high_score_mask, :]
        max_score = max_score[high_score_mask]
        label     = label[high_score_mask]

        # Decode reg_pred(delta) by using anchor's geometry
        bboxes = self._decode(anchor_det, reg_pred, hdg_pred)

        # Clip 2d bbox's boundary if exceed image.shape, TODO, I disable clipper temply 
        # if img_batch is not None:
        #     bboxes = self.clipper(bboxes, img_batch)

        # Non-Maximum Suppresion
        
        if cls_agnostic:
            keep_inds = nms(bboxes[:, :4], max_score, nms_iou_thr)
        else:
            max_coordinate = bboxes.max()
            nms_bbox = bboxes[:, :4] + label.float().unsqueeze() * (max_coordinate)
            keep_inds = nms(nms_bbox, max_score, nms_iou_thr)

        # Filter Box by NMS result
        # print(f"bboxes before nms = {bboxes.shape}") # [1644, 11]
        bboxes    = bboxes   [keep_inds]
        max_score = max_score[keep_inds]
        label     = label    [keep_inds]
        # print(f"bboxes after nms = {bboxes.shape}") # [1355, 11]
        
        # print(f"Number of detection = {bboxes.shape[0]}")
        # Post Optimization
        if is_post_opt:
            max_score, bboxes, label = self._post_process(max_score, bboxes, label, P2s)
        return max_score, bboxes, label

    def loss(self, cls_preds, reg_preds, anchors_dummy, annotations, P2s):
        '''
        Input: 
            * B: Batch size 
            * N_A: Number of anchor
            cls_preds: [B, N_A, 2]
            reg_preds: [B, N_A, 12]
                cls_preds and reg_preds are predicted value of the netowrk
            annotations: [B, max_lenght_label, 12]

        Variable Explain:
        n_pos - integer - 40 -
            Number of positives. How many anchor are postive sample(assgin to an gt)
        
        n_neg - integer - 3720
            Number of negatives. How many anchor are negative sample
        
        n_gts - [] (was number_of_positives)
            Number of ground trues
        
        pos_inds - tensor - [n_pos] - [40]
            index of anchors that be assigned as postives 
                index of anchors that be assigned as postives 
            index of anchors that be assigned as postives 
        
        neg_inds - tensor - [n_neg] = [3720]
            index of anchors that be assigned as nagative 
                index of anchors that be assigned as nagative 
            index of anchors that be assigned as nagative 
        
        reg_targets - tensor - [n_pos, 12] - [40, 12] - (was pos_bbox_targets)
            target value for postive anchors; network need to predict these 12 values
            Should output by _encode()
        
        hdg_targets - tensor - [n_pos, 1]  - [40, 1] - (was targets_alpha_cls)
            target class for positive anchors;
            should be output by _encode()
        
        labels - tensor - [Number of anchor, Total classes] - [3860, 1]
            if entry == 1 means it's postive anchor for this class
            if entry == 0 means it's a negative anchor for it
            if entry == -1 means need to ignore this anchor
        '''
        
        cls_loss = []
        reg_loss = []

        for j in range(cls_preds.shape[0]): # For every batch
            # Get prediction 
            reg_pred = reg_preds[j]
            cls_pred = cls_preds[j][..., 0:self.num_classes]
            hdg_pred = cls_preds[j][..., self.num_classes:self.num_classes+1] # Predict heading of the object (front or back)

            # only select useful bbox_annotations, TODO Can I make use of "Dont care" as hard-nagative sample?
            anno_j = annotations[j, :, :] # [4, 12]
            anno_j = anno_j[ anno_j[:, 4] != -1 ] # [4, 12], [3, 12]# filter out cls_idx == -1
            n_anno = anno_j.shape[0]

            if len(anno_j) == 0: # if no ground true
                cls_loss.append(torch.tensor(0).cuda().float())
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
                continue
            
            labels = self.anchors.new_full((self.n_anchor, self.num_classes),
                                       -1, # fill tensor with -1. -1 means not computed, binary for each class
                                       dtype=torch.float)
            
            #########################
            ### Anchor Assignment ###
            #########################
            '''
            anchor_assignment, [14284]
                record gt assignment of every anchor 
                -2 means doesn't assign, ignore this anchor
                -1 means it's a negative sameple
                 0~self.num_class-1 means it's belong to that gt
            '''
            if self.anchor_assignment_method == 'maxIoU':
                # Convert label to tensor 
                IoU = calc_iou(self.anchors[:, :4], anno_j[:, :4]) # IoU = [14284, 4]
                # Find max overlap groundture with the anchor
                iou_max, iou_argmax = IoU.max(dim=1) # [14284], [14284]
                anchor_assignment = iou_argmax.new_full((iou_argmax.shape[0], ), -2, dtype=torch.long)
                # Assign positive anchor
                anchor_assignment[iou_max >= self.loss_cfg.fg_iou_threshold] = iou_argmax[iou_max >= self.loss_cfg.fg_iou_threshold]
                # Assign negative anchor
                anchor_assignment[iou_max <  self.loss_cfg.bg_iou_threshold] = -1

            elif self.anchor_assignment_method == '3Ddistance':
                # Anchor assignment by choosing closest anchor 
                N_ANCHOR_ASSIGN_TO_GT = 4
                
                anchor_assignment = self.anchors.new_full((self.n_anchor, ), -2, dtype=torch.long)
                anno_idx     = self.anchors.new_full((3, n_anno, self.n_anchor), 0)
                norm1_tensor = self.anchors.new_full((   n_anno, self.n_anchor), 0)

                for i in range(n_anno):# Can this be accelarated by tensor operation? 
                    anno_idx[0, i, :] = anno_j[i, 12].repeat( self.anchors.shape[0] )
                    anno_idx[1, i, :] = anno_j[i, 14].repeat( self.anchors.shape[0] )
                    anno_idx[2, i, :] = anno_j[i, 15].repeat( self.anchors.shape[0] )
                
                for i in range(n_anno):
                    # Calcualte Norm
                    norm1_tensor[i, :] = abs(self.anchors[:, 12] - anno_idx[0, i, :] ) + abs(self.anchors[:, 14] - anno_idx[1, i, :])
                    
                    # Don't use anchor that heading is in wrong direction
                    norm1_tensor[i, torch.abs(torch.cos(self.anchors[:, 15] - anno_idx[2, i, :])) < pi/4 ] = float('inf')

                # Potential Issue: What if two ground true sharing one anchor?
                # This might happened in crowd scene
                _, topk_argmin = torch.topk(norm1_tensor, N_ANCHOR_ASSIGN_TO_GT, dim = 1, largest=False)

                # Assign positive anchor
                for i in range(n_anno):
                    anchor_assignment[ topk_argmin[i, :] ] = i
                
                # Assign negative anchor
                anchor_assignment[ anchor_assignment < 0 ] = -1
            
            elif self.anchor_assignment_method == 'L1distance':
            
                anchor_assignment = self.anchors.new_full((self.n_anchor, ), -1, dtype=torch.long)
                dist_tensor = self.anchors.new_full((self.n_anchor, ), 0)
                for i in range(n_anno):
                    # Calcualte Distance
                    dist_tensor = abs(self.anchors[:, 5] - anno_j[i, 5] ) + abs(self.anchors[:, 6] - anno_j[i, 6])

                    # Don't use anchor that heading is in wrong direction
                    dist_tensor[torch.abs(torch.cos(self.anchors[:, 15] - anno_j[i, 15])) < np.pi/4 ] = float('inf')

                    # Assign positive anchor, Don't use anchor that too far from z3d
                    anchor_assignment[ torch.logical_and( dist_tensor <= 16*2, torch.abs(self.anchors[:, 7] - anno_j[i, 7]) < 2) ] = i

                    # Assign ignore anchor 
                    anchor_assignment[ torch.logical_and( dist_tensor <= 16*2, torch.abs(self.anchors[:, 7] - anno_j[i, 7]) >= 2)] = -2
                
                    # Assign negative anchor
                    # anchor_assignment[ anchor_assignment < 0 ] = -1

                ign_inds = torch.nonzero(anchor_assignment ==-2, as_tuple=False).squeeze(dim=1)
                n_ign = ign_inds.shape[0]
            else:
                assert False, f"Can't find anchor assignment method: {self.anchor_assignment_method}"
            # Get index of postive and negative anchor
            pos_inds = torch.nonzero(anchor_assignment >= 0, as_tuple=False).squeeze(dim=1)
            neg_inds = torch.nonzero(anchor_assignment ==-1, as_tuple=False).squeeze(dim=1)
            # 
            n_pos = pos_inds.shape[0]
            n_neg = neg_inds.shape[0]
            # 
            # print(f"n_pos, n_neg = {(n_pos, n_neg)}") # (86, 14078)

            if len(pos_inds) > 0:
                
                # Encode (groundtrue - anchor) and get target values 
                reg_targets, hdg_targets = self._encode(self.anchors[pos_inds], anno_j[anchor_assignment[pos_inds], :])
                
                # Assign class in labels
                labels[pos_inds, :] = 0 # negative sample for other classes 
                labels[pos_inds, anno_j[anchor_assignment[pos_inds], 4].long() ] = 1 # postive sample for this class

                # Calculate regression loss and heading loss, this is the first time prediction result been used
                
                # print(f"reg_targets = {reg_targets}")
                # print(f"reg_pred[pos_inds] = {reg_pred[pos_inds]}")
                
                reg_loss_j = self.reg_loss(reg_targets, reg_pred[pos_inds])
                hdg_loss_j = self.hdg_loss(hdg_pred[pos_inds], hdg_targets) # This order is important, don't change it                
                # Weighted loss
                loss_weighted = torch.cat([reg_loss_j, hdg_loss_j], dim=1) * self.regression_weight #[N, 13]

                reg_loss.append(loss_weighted.mean(dim=0)) #[13]
                    
            else: # if len(pos_inds) == 0: # TODO how to assign negative sample
                reg_loss.append(reg_preds.new_zeros(self.num_regression_loss_terms))
            
            # Assign negative sample in labels
            if len(neg_inds) > 0: labels[neg_inds, :] = 0 # Negative sample
            # 
            # if len(ign_inds) > 0: labels[ign_inds, :] = -1 # Ignore sample

            # 
            # print(f"Number of (pos, neg) in cls_pred = {(torch.numel(cls_pred[ cls_pred > 0 ]), torch.numel(cls_pred[ cls_pred < 0 ]))}")
            # print(f"Mean of positive = {cls_pred[ cls_pred > 0 ].mean()}")
            # Classification Loss
            # print(f"self.cls_loss(cls_pred, labels).sum() = {self.cls_loss(cls_pred, labels).sum()}")
            # print(f"(len(pos_inds) + len(neg_inds)) = {(len(pos_inds) + len(neg_inds))}")
            
            cls_loss.append(self.cls_loss(cls_pred, labels).sum() / (len(pos_inds) + len(neg_inds)))
            # print(f"labels.shape = {labels.shape}") # [9332, 1]
            # unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
            # anchor_assign = dict(zip(unique, counts)) # {0.0: 9308, 1.0: 24}


        # print(f"cls_loss.shape = {len(cls_loss)}") # [8, ..]
        # print(f"reg_loss.shape = {len(reg_loss)}") # [8, ..]
        
        # Get classify loss and regression loss
        cls_loss = torch.stack(cls_loss).mean(dim=0, keepdim=True)
        reg_loss = torch.stack(reg_loss, dim=0) #[B, 12]
        
        # Weighted regression loss by number of ground true in each image
        n_gts = [torch.sum(annotations[j, :, 4] >= 0) for j in range(annotations.shape[0])]
        weights = reg_pred.new(n_gts).unsqueeze(1) #[B, 1] - [[4]]
        weighted_regression_losses = torch.sum(weights * reg_loss / (torch.sum(weights) + 1e-6), dim=0)
        reg_loss = weighted_regression_losses.mean(dim=0, keepdim=True)
        
        log_dict = {}
        log_dict['1/cls_loss'] = cls_loss
        log_dict['1/reg_loss'] = reg_loss
        log_dict['1/total_loss'] = cls_loss + reg_loss
        log_dict['2/dx']     = weighted_regression_losses[0]
        log_dict['2/dy']     = weighted_regression_losses[1]
        log_dict['2/dw']     = weighted_regression_losses[2]
        log_dict['2/dh']     = weighted_regression_losses[3]
        log_dict['2/cdx']    = weighted_regression_losses[4]
        log_dict['2/cdy']    = weighted_regression_losses[5]
        log_dict['2/cdz']    = weighted_regression_losses[6]
        log_dict['4/d_sin']  = weighted_regression_losses[7]
        log_dict['4/d_cos']  = weighted_regression_losses[8]
        log_dict['4/dw']     = weighted_regression_losses[9]
        log_dict['4/dh']     = weighted_regression_losses[10]
        log_dict['4/dl']     = weighted_regression_losses[11]
        log_dict['4/d_hdg']  = weighted_regression_losses[12]
        log_dict['3/n_positive_anchor'] = torch.tensor([n_pos])
        log_dict['3/n_negative_anchor'] = torch.tensor([n_neg])
        # log_dict['3/n_ignored_anchor']  = torch.tensor([n_ign])
        log_dict['3/n_positive_predict'] = torch.numel(cls_pred[ cls_pred >  0 ])
        log_dict['3/n_negative_predict'] = torch.numel(cls_pred[ cls_pred <= 0 ])
        log_dict['3/mean_positive_predict'] = cls_pred[ cls_pred >  0 ].mean()
        log_dict['3/mean_negative_predict'] = cls_pred[ cls_pred <= 0 ].mean()

        return log_dict
