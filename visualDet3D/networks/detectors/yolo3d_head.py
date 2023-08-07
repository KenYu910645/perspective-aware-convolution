import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

from visualDet3D.networks.detectors.losses import CIoULoss, DIoULoss, SigmoidFocalLoss, ModifiedSmoothL1Loss, IoULoss
from visualDet3D.networks.detectors.anchors import Anchors
from visualDet3D.utils.cal import calc_iou, BackProjection, ClipBoxes
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.lib.look_ground import LookGround
from visualDet3D.networks.lib.cbam import CBAM
from visualDet3D.networks.lib.bam import BAM
from visualDet3D.networks.lib.coordatten import CoordAtt
from visualDet3D.networks.lib.dcn import DeformableConv2d
from visualDet3D.networks.lib.rfb import BasicRFB
from visualDet3D.networks.lib.aspp import ASPP
from visualDet3D.networks.lib.pac_module import PAC_module, PerspectiveConv2d
from visualDet3D.networks.lib.das import DepthAwareSample

class Yolo3D_Head(nn.Module):
    def __init__(self, cfg):
        super(Yolo3D_Head, self).__init__()
        
        self.cfg = cfg
        self.num_classes = cfg.detector.head.num_classes

        self.anchors = Anchors(cfg, is_training_process = True)

        self.build_loss()

        self.init_layers(cfg.detector.head)

        self.backprojector = BackProjection()

        self.clipper = ClipBoxes()
        
        # For Anchor staticical
        self.n_miss_gt = 0
        self.n_cover_gt = 0
        self.n_assign_anchor = 0

    def init_layers(self, head_cfg):
        
        a_module_cfg = self.cfg.detector.attention_module
        d_module_cfg = self.cfg.detector.dilation_module
        depth_cfg    = self.cfg.detector.depth_branch 

        ####################################
        ### Attention Module Declaration ###
        ####################################
        if a_module_cfg.use_spatial_attention or a_module_cfg.use_channel_attention:
            self.cbam_layer = CBAM(head_cfg.num_features_in, 
                                    reduction_ratio = 16, 
                                    pool_types = ['avg', 'max'], 
                                    use_spatial = self.use_spatial_attention, 
                                    use_channel = self.use_channel_attention)
            print(f"CBAM attention module = {self.cbam_layer}")
        
        if a_module_cfg.use_bam:
            self.bam_layer = BAM(head_cfg.num_features_in)
            print(f"BAM attention module = {self.bam_layer}")
        
        if a_module_cfg.use_coordinate_attetion:
            self.coord_atten_layer = CoordAtt(head_cfg.num_features_in, head_cfg.num_features_in)
            print(f"Coordinate attention module = {self.coord_atten_layer}")
        
        ###################################
        ### Dilation Module Declaration ###
        ###################################
        if d_module_cfg.num_dcnv2 != 0:
            self.dcn_layers = nn.ModuleList([
                DeformableConv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
                for _ in range(head_cfg.num_dcnv2)
            ])

        if d_module_cfg.num_pac_layer != 0:
            self.pac_layers = []
            for _ in range(d_module_cfg.num_pac_layer):
                self.pac_layers.append( nn.Sequential(
                    PerspectiveConv2d(head_cfg.num_features_in,
                                      head_cfg.num_features_in,
                                      input_shape = (18, 80),
                                      d_rate_xy = d_module_cfg.d_rate_xy,
                                      lock_theta_ortho=d_module_cfg.lock_theta_ortho),
                    nn.BatchNorm2d(head_cfg.num_features_in),
                    nn.ReLU(),) )
            self.pac_layers = [n.to("cuda") for n in d_module_cfg.pac_layers] # TODO, tmp
            print(f"pac_layers = {self.pac_layers}")
        
        if d_module_cfg.is_pac_module:
            self.pac_layers = PAC_module(head_cfg.num_features_in, head_cfg.num_features_in, lock_theta_ortho=d_module_cfg.lock_theta_ortho)
            print(f"PAC_module = {self.pac_layers}")
            
        if d_module_cfg.is_rfb:
            self.rfb_layer = BasicRFB(1024, 1024, scale = 1.0, visual=2)
            print(f"Using RFB Moduel = {self.rfb_layer}")

        if d_module_cfg.is_aspp:
            self.aspp_layer = ASPP(1024, 1024)
            print(f"Using ASPP Moduel = {self.aspp_layer}")
        
        #########################################
        ### Classification Branch Declaration ###
        #########################################
        if head_cfg.is_das:
            self.cls_feature_extraction = nn.Sequential(
                nn.Conv2d(head_cfg.num_features_in, head_cfg.cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_cfg.cls_feature_size, head_cfg.cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                DepthAwareSample(head_cfg.cls_feature_size, head_cfg.num_cls_output*head_cfg.num_anchors, head_cfg.num_cls_output, kernel_size=3, padding=1),
            )
        else:
            self.cls_feature_extraction = nn.Sequential(
                nn.Conv2d(head_cfg.num_features_in, head_cfg.cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_cfg.cls_feature_size, head_cfg.cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_cfg.cls_feature_size, head_cfg.num_anchors*head_cfg.num_cls_output, kernel_size=3, padding=1),
                AnchorFlatten(head_cfg.num_cls_output)
            )
            self.cls_feature_extraction[-2].weight.data.fill_(0)
            self.cls_feature_extraction[-2].bias.data.fill_(0)
        
        #####################################
        ### Regression Branch Declaration ###
        #####################################
        if head_cfg.is_das:
            self.reg_feature_extraction = nn.Sequential(
                LookGround(head_cfg.num_features_in),
                nn.Conv2d(head_cfg.num_features_in, head_cfg.reg_feature_size, 3, padding=1),
                nn.BatchNorm2d(head_cfg.reg_feature_size),
                nn.ReLU(),
                nn.Conv2d(head_cfg.reg_feature_size, head_cfg.reg_feature_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(head_cfg.reg_feature_size),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                DepthAwareSample(head_cfg.reg_feature_size, head_cfg.num_anchors*head_cfg.num_reg_output, head_cfg.num_reg_output, kernel_size=3, padding=1),
            )
        else:
            self.reg_feature_extraction = nn.Sequential(
                LookGround(head_cfg.num_features_in),
                nn.Conv2d(head_cfg.num_features_in, head_cfg.reg_feature_size, 3, padding=1),
                nn.BatchNorm2d(head_cfg.reg_feature_size),
                nn.ReLU(),
                nn.Conv2d(head_cfg.reg_feature_size, head_cfg.reg_feature_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(head_cfg.reg_feature_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_cfg.reg_feature_size, head_cfg.num_anchors*head_cfg.num_reg_output, kernel_size=3, padding=1),
                AnchorFlatten((head_cfg.num_reg_output))
            )
            self.reg_feature_extraction[-2].weight.data.fill_(0)
            self.reg_feature_extraction[-2].bias.data.fill_(0)

        ################################
        ### Depth Branch Declaration ###
        ################################
        if depth_cfg.is_seperate_cz:
            if depth_cfg.cz_pred_mode == "fc":
                self.cz_feature_extraction = nn.Sequential( # Reduce dimension from 1024 to 256
                    nn.Conv2d(head_cfg.num_features_in, 256, kernel_size = 1),
                )
                self.cz_fc_layer = nn.Sequential(
                    nn.Linear(18*80*256, 1440),
                    nn.ReLU(),
                    nn.Linear(1440 , 1440),
                )
            elif depth_cfg.cz_pred_mode == "look_ground":
                self.cz_feature_extraction = nn.Sequential(
                    LookGround(head_cfg.num_features_in),
                    nn.Conv2d(head_cfg.num_features_in, depth_cfg.cz_reg_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(head_cfg.cz_reg_dim),
                    nn.ReLU(),
                    nn.Conv2d(depth_cfg.cz_reg_dim, depth_cfg.cz_reg_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(depth_cfg.cz_reg_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(depth_cfg.cz_reg_dim, head_cfg.num_anchors*1, kernel_size=3, padding=1),
                    AnchorFlatten((1)),
                )
                self.cz_feature_extraction[-2].weight.data.fill_(0)
                self.cz_feature_extraction[-2].bias.data.fill_(0)
            
            elif depth_cfg.cz_pred_mode == "oridinal_loss": # TODO
                raise NotImplementedError
    
    def forward(self, inputs):
        
        a_module_cfg = self.cfg.detector.attention_module
        d_module_cfg = self.cfg.detector.dilation_module
        depth_cfg    = self.cfg.detector.depth_branch 

        ########################
        ### Attention Module ###
        ########################
        if a_module_cfg.use_spatial_attention or a_module_cfg.use_channel_attention:
            inputs['features'] = self.cbam_layer(inputs['features']) # [1024, 18, 80]
        
        if a_module_cfg.use_bam:
            inputs['features'] = self.bam_layer(inputs['features'])
        
        if a_module_cfg.use_coordinate_attetion:
            inputs['features'] = self.coord_atten_layer(inputs['features'])
        
        ########################
        ### Dilation  Module ###
        ########################
        if d_module_cfg.num_dcnv2 != 0:
            for i in range(d_module_cfg.num_dcnv2):
                inputs['features'] = self.dcn_layers[i](inputs['features'])
        
        if d_module_cfg.num_pac_layer != 0:
            for i in range(d_module_cfg.num_pac_layer):
                inputs['features'] = self.pac_layers[i](inputs)
        
        if d_module_cfg.is_pac_module:
            inputs['features'] = self.pac_layers(inputs)

        if d_module_cfg.is_rfb:
            inputs['features'] = self.rfb_layer(inputs['features'])
        
        if d_module_cfg.is_aspp:
            inputs['features'] = self.aspp_layer(inputs['features'])
        
        cls_preds = self.cls_feature_extraction(inputs['features'])
        reg_preds = self.reg_feature_extraction(inputs)
        
        ################################
        ### Depth Prediction  Module ###
        ################################
        if depth_cfg.is_seperate_cz:
            if depth_cfg.cz_pred_mode == "fc":
                dep_preds = self.cz_feature_extraction(inputs['features'])
                dep_preds = dep_preds.permute(0, 2, 3, 1)
                dep_preds = dep_preds.contiguous().view(dep_preds.shape[0], -1) # torch.Size([8, 368640])
                dep_preds = self.cz_fc_layer(dep_preds) # torch.Size([8, 1440])
                dep_preds = dep_preds.view(dep_preds.shape[0], -1, 18, 80) # torch.Size([8, 1, 18, 80])
                dep_preds = dep_preds.repeat(1, 32, 1, 1) # torch.Size([8, 32, 18, 80])
            
            elif depth_cfg.cz_pred_mode == "look_ground":
                dep_preds = self.cz_feature_extraction(inputs)
            
            elif depth_cfg.cz_pred_mode == "oridinal_loss":
                raise NotImplementedError
        else:
            dep_preds = None
        
        return {"cls_preds" : cls_preds,
                "reg_preds" : reg_preds,
                "dep_preds" : dep_preds,}

    def build_loss(self):
        loss_cfg = self.cfg.detector.loss
        
        self.register_buffer("balance_weights", torch.tensor(loss_cfg.balance_weight, dtype = torch.float32))
        self.focal_loss = SigmoidFocalLoss(gamma = loss_cfg.focal_loss_gamma, balance_weights = self.balance_weights)
        self.smooth_L1_loss = ModifiedSmoothL1Loss(loss_cfg.L1_regression_alpha)

        if   loss_cfg.iou_type == "iou" : self.loss_iou =  IoULoss ()
        elif loss_cfg.iou_type == "diou": self.loss_iou =  DIoULoss()
        elif loss_cfg.iou_type == "ciou": self.loss_iou =  CIoULoss()
        # else:
        #     self.loss_iou = IoULoss()

        # regression_weight = kwargs.get("regression_weight", [1 for _ in range(self.cfg.detector.num_regression_loss_terms)]) #default 12 only use in 3D
        self.register_buffer("regression_weight", torch.tensor(loss_cfg.regression_weight, dtype=torch.float))

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

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

        # unique, counts = np.unique(argmax_overlaps.cpu().numpy(), return_counts=True)
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
        
        # print(f"pos_2dbox = {pos_2dbox.shape}") # [40, 4]
        # print(f"pos_anno = {pos_anno.shape}") # [40, 12]
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
        targets_dx = (gx - px) / pw
        targets_dy = (gy - py) / ph
        targets_dw = torch.log(gw / pw)
        targets_dh = torch.log(gh / ph)

        # 3D bbox center on image plane
        targets_cdx = (sampled_gt_bboxes[:, 5] - px) / pw
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

        hdg_target = (torch.cos(sampled_gt_bboxes[:, 11:12]) > 0).float()
        return targets, hdg_target #[N, 4]

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
            pos_anno = gt_bboxes.new_zeros([0, 4])
        else:
            pos_anno = gt_bboxes[pos_assigned_gt_inds[pos_inds], :]
        
        return_dict = dict(
            pos_inds = pos_inds,
            neg_inds = neg_inds,
            pos_2dbox = anchors[pos_inds],
            neg_2dbox = anchors[neg_inds],
            pos_anno = pos_anno,
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
        anchors, useful_mask, anchor_mean_std = self.anchors(img_batch, P2)
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
            cls_preds - [B, N_A, 2] - [1, 46080, 2]
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
        test_cfg = self.cfg.detector.test

        cls_preds = preds['cls_preds'] # [1, 46080, 2]
        reg_preds = preds['reg_preds'] # [1, 46080, 12]
        dep_preds = preds['dep_preds']
        
        assert cls_preds.shape[0] == 1 # batch == 1

        cls_preds= cls_preds.sigmoid()
        cls_pred = cls_preds[0][..., 0:self.num_classes]
        hdg_pred = cls_preds[0][..., self.num_classes:self.num_classes+1]
        reg_pred = reg_preds[0]
        if self.cfg.detector.depth_branch.is_seperate_cz: dep_preds = dep_preds[0]
        
        ank_2dbbox = anchors['anchors'][0] #[N, 4]
        ank_zscwhl = anchors['anchor_mean_std_3d'] #[N, K, 2]
        useful_mask = anchors['mask'][0] #[N, ]
        
        ank_2dbbox = ank_2dbbox[useful_mask]
        ank_zscwhl = ank_zscwhl[useful_mask]
        cls_pred   = cls_pred  [useful_mask]
        hdg_pred   = hdg_pred  [useful_mask]
        reg_pred   = reg_pred  [useful_mask]
        if self.cfg.detector.depth_branch.is_seperate_cz: dep_preds = dep_preds[useful_mask]

        # Find highest score in all classes
        max_score, label = cls_pred.max(dim=-1) 
        high_score_mask = (max_score > test_cfg.score_thr)
        #
        ank_2dbbox = ank_2dbbox[high_score_mask, :]
        ank_zscwhl = ank_zscwhl[high_score_mask, :]
        cls_pred   = cls_pred  [high_score_mask, :]
        hdg_pred   = hdg_pred  [high_score_mask, :]
        reg_pred   = reg_pred  [high_score_mask, :] # [n_det, 12/20]
        max_score  = max_score [high_score_mask]
        label      = label     [high_score_mask]
        if self.cfg.detector.depth_branch.is_seperate_cz: dep_preds = dep_preds[high_score_mask]
        #
        bboxes, mask = self._decode(ank_2dbbox, reg_pred, ank_zscwhl, label, hdg_pred, cz_deltas=dep_preds)
        
        # Clip 2d bbox's boundary if exceed image.shape
        if img_batch is not None:
            bboxes = self.clipper(bboxes, img_batch)
        
        cls_pred  = cls_pred [mask]
        max_score = max_score[mask]
        bboxes    = bboxes   [mask]

        ###############################
        ### Non-Maxiumum Supression ###
        ###############################
        # print(f"bboxes before nms = {bboxes.shape}") # [184, 11]
        if test_cfg.cls_agnostic:
            keep_inds = nms(bboxes[:, :4], max_score, test_cfg.nms_iou_thr)
        else:
            max_coordinate = bboxes.max()
            nms_bbox = bboxes[:, :4] + label.float().unsqueeze() * (max_coordinate)
            keep_inds = nms(nms_bbox, max_score, test_cfg.nms_iou_thr)

        bboxes      = bboxes   [keep_inds]
        max_score   = max_score[keep_inds]
        label       = label    [keep_inds]
        # print(f"bboxes after nms = {bboxes.shape}") # [1, 11]

        ####################
        ### Post Process ###
        ####################
        if test_cfg.post_optimization:
            max_score, bboxes, label = self._post_process(max_score, bboxes, label, P2s)
        # print(bboxes.shape) # [n_det, 11] # [x1, y1, x2, y2, cx, cy, cz, w, h, l, alpha]
        
        return max_score, bboxes, label

    def loss(self, preds, anchors, annos, P2s):
        '''
        Input:
            preds is the preds result of the network:
                preds['cls_preds']  : [B, num_anchor, 2]
                preds['reg_preds']  : [B, num_anchor, 12]
                preds['dep_preds']   : [B, num_anchor, 1] or None 
                preds['nom_preds'] : [B, num_anchor, 8] or None
            anchors
                anchors['anchors'] : [1, num_anchor, 4]
                anchors['anchor_mean_std_3d'] : [num_anchor, 1, 6, 2]
                # [z,  sinalpha, cosalpha, w, h, l]
            annos: [B, max_lenght_label, 16], annotataions are ground trues # [8, 12, 16]
                [x1, y1, x2, y2, cls_index, cx, cy, z, w, h, l, alpha, x3d, y3d, z3d, rot_y]
        '''
        # cls_preds and reg_preds are predicted by netowrk 
        cls_preds = preds['cls_preds'] # [B, num_anchor, 2]
        reg_preds = preds['reg_preds'] # [B, num_anchor, 12]
        dep_preds = preds['dep_preds'] # [B, num_anchor, 1] or None 

        # annotataions are ground trues
        ank_2dbbox = anchors['anchors'][0] # [num_anchor, 4]
        ank_zscwhl = anchors['anchor_mean_std_3d'] # [num_anchor, 1, 6, 2]
        
        cls_loss = []
        reg_loss = []
        iou_loss = []
        dep_loss = []
        n_pos_list = []
        for j in range(cls_preds.shape[0]): # j is batch index
            
            reg_pred = reg_preds[j] # [B, 12/20]
            cls_pred = cls_preds[j][..., 0:self.num_classes] # [B, 1]
            hdg_pred = cls_preds[j][..., self.num_classes:self.num_classes+1] # [B, 1], Predict heading of the object (front or back)
            if dep_preds != None: dep_pred = dep_preds[j]
            anno_j = annos[j, :, :] # [n_gts, 16/24]
            
            # Filter anchors by mask
            useful_mask = anchors['mask'][j] #[N]
            ank_2dbbox_j         = ank_2dbbox         [useful_mask]
            ank_zscwhl_j         = ank_zscwhl         [useful_mask]
            reg_pred             = reg_pred           [useful_mask]
            cls_pred             = cls_pred           [useful_mask]
            hdg_pred             = hdg_pred           [useful_mask]
            if dep_preds != None: dep_pred = dep_pred [useful_mask]
            
            # Use only useful annotation
            anno_j = anno_j[anno_j[:, 4] != -1]
            
            if len(anno_j) == 0: # If this image doesn't contain any ground true, just skip it
                cls_loss.append(torch.tensor(0).cuda().float())
                reg_loss.append(reg_preds.new_zeros(self.cfg.detector.head.num_regression_loss_terms))
                iou_loss.append(reg_preds.new_zeros(1)[0])
                dep_loss.append(reg_preds.new_zeros(1))
                n_pos_list.append(0)
                continue
            
            # This is like retinanet's and YOLO's MaxIouAssignerm 
            assign_dict = self._assign(ank_2dbbox_j, anno_j, **self.cfg.detector.loss) # doesn't involve prediction
            # print(f"assign_dict['num_gt'] = {assign_dict['num_gt']}") # 4 
            # print(f"assign_dict['assigned_gt_inds'] = {assign_dict['assigned_gt_inds'].shape}") # [7548]
            # print(assign_dict['assigned_gt_inds'].unique()) # -1,  0,  1, 0 means negative, 1 meams positive, 
            # print(f"assign_dict['max_overlaps'] = {assign_dict['max_overlaps'].shape}") # [7548]
            # print(f"assign_dict['labels'] = {assign_dict['labels'].shape}") # [7548]
            
            # This is for checking GAC's anchor'a miss rate
            # unique, counts = np.unique(assign_dict['assigned_gt_inds'].cpu().numpy(), return_counts=True)
            # anchor_assign = dict(zip(unique, counts)) # {-1: 100, 0: 3720, 1: 40}
            
            sample_dict    = self._sample(assign_dict, ank_2dbbox_j, anno_j)
            # sample_dict['pos_inds']: [n_pos]
            # sample_dict['neg_inds']: [n_neg]
            # sample_dict['pos_2dbox']: [n_pos, 4]
            # sample_dict['neg_2dbox']: [n_neg, 4]
            # sample_dict['pos_anno']: [n_pos, 16/24]
            # sample_dict['pos_assigned_gt_inds']: [n_pos]

            labels = ank_2dbbox_j.new_full((ank_2dbbox_j.shape[0], self.num_classes),
                                    -1, # -1 not computed, binary for each class
                                    dtype=torch.float)

            pos_inds = sample_dict['pos_inds']
            neg_inds = sample_dict['neg_inds']
            #
            n_pos = pos_inds.shape[0]
            n_neg = neg_inds.shape[0]
            # print(f"n_pos, n_neg = {(n_pos, n_neg)}") # (86, 14078) n_pos, n_neg = (40, 3720)

            if len(pos_inds) > 0:
                pos_assigned_gt_label = anno_j[sample_dict['pos_assigned_gt_inds'], 4].long()
                
                selected_mask, selected_anchor_3d = self._get_anchor_3d(
                    sample_dict['pos_2dbox'],
                    ank_zscwhl_j[pos_inds],
                    pos_assigned_gt_label,
                )
                if len(selected_anchor_3d) > 0:
                    
                    # Filter out anchor that z < 0 
                    pos_inds  = pos_inds[selected_mask]
                    pos_2dbox = sample_dict['pos_2dbox'][selected_mask]
                    pos_anno  = sample_dict['pos_anno'][selected_mask]
                    # pos_assigned_gt = sample_dict['pos_assigned_gt_inds'][selected_mask]
                    
                    # Output anchor for debugging
                    # import pickle
                    # torch.set_printoptions(threshold=10_000)
                    # with open("GAC_head_anchor_2D.pkl", 'wb') as f:
                    #     pickle.dump(pos_2dbox.detach().cpu().numpy(), f)
                    # with open("GAC_head_anchor_3D.pkl", 'wb') as f:
                    #     pickle.dump(selected_anchor_3d.detach().cpu().numpy(), f)
                    # print(f"Output anchor's information to GAC_head_anchor_2D.pkl and GAC_head_anchor_3D.pkl")

                    ##############
                    ### Encode ###
                    ##############
                    reg_target, hdg_target = self._encode(pos_2dbox, pos_anno, selected_anchor_3d)
                    # print(f"reg_target = {reg_target.shape}") # [n_pos, 12]
                    # print(f"hdg_target = {hdg_target.shape}") # [n_pos, 1]
                    
                    label_index = pos_assigned_gt_label[selected_mask]
                    labels[pos_inds, :] = 0
                    labels[pos_inds, label_index] = 1

                    ######################
                    ### Get Depth Loss ###
                    ######################
                    if self.cfg.detector.depth_branch.is_seperate_cz:
                        dep_target = reg_target[:, 6:6+1] # [697, 1] 
                        if self.cfg.detector.depth_branch.cz_pred_mode == "look_ground" or self.cfg.detector.depth_branch.cz_pred_mode == "fc":
                            dep_loss_j = self.smooth_L1_loss(dep_target, dep_pred[pos_inds]) # [141, 1]
                        
                        elif self.cfg.detector.depth_branch.cz_pred_mode == "oridinal_loss":
                            raise NotImplementedError
                        
                        # print(f"dep_loss_j.requires_grad = {dep_loss_j.requires_grad}")
                        dep_loss.append( dep_loss_j.mean(dim=0) * 3) # TODO, this 3 ratio is derived from regersssion_weight
                        # 
                        reg_target = torch.cat((reg_target[:, :6], reg_target[:, 6+1:]), dim=1) # [694, 11]
                    
                    ###########################
                    ### Get Regression Loss ###
                    ###########################
                    # print(f"reg_target = {reg_target.shape}") # [628, 12]
                    # print(f"reg_pred[pos_inds] = {reg_pred[pos_inds].shape}") # [628, 20]
                    if self.cfg.detector.loss.iou_type == "baseline":
                        reg_loss_j = self.smooth_L1_loss(reg_target, reg_pred[pos_inds]) # This is the first time, loss() used prediction result
                    else: # Disable L1 2dbbox
                        reg_loss_j = self.smooth_L1_loss(reg_target[:, 4:], reg_pred[pos_inds][:, 4:])
                    hdg_loss_j = self.bce_loss      (hdg_pred[pos_inds], hdg_target)
                    reg_loss.append( (torch.cat([reg_loss_j, hdg_loss_j            ], dim=1)*self.regression_weight).mean(dim=0) )
                
                    ####################
                    ### Get IOU Loss ###
                    ####################
                    if self.cfg.detector.loss.iou_type != "baseline":
                        pos_prediction_decoded, mask = self._decode(pos_2dbox, reg_pred[pos_inds],  ank_zscwhl_j[pos_inds], label_index, hdg_pred[pos_inds])
                        pos_target_decoded    , _    = self._decode(pos_2dbox, reg_target        ,  ank_zscwhl_j[pos_inds], label_index, hdg_pred[pos_inds])
                        loss_iou_j = self.loss_iou(pos_prediction_decoded[mask, :4], pos_target_decoded[mask, :4])
                        iou_loss.append(loss_iou_j.mean(dim=0))
                    
                    n_pos_list.append(anno_j.shape[0])
            else:
                reg_loss.append(reg_preds.new_zeros(self.cfg.detector.head.num_regression_loss_terms))
                n_pos_list.append(anno_j.shape[0])
                if self.cfg.detector.loss.iou_type != "baseline": iou_loss.append(reg_preds.new_zeros(1)[0])
                if dep_preds != None: dep_loss.append(reg_preds.new_zeros(1))

            if len(neg_inds) > 0:
                labels[neg_inds, :] = 0
            
            # Get classification loss
            cls_loss.append(self.focal_loss(cls_pred, labels).sum() / (len(pos_inds) + len(neg_inds)))
        
        cls_loss = torch.stack(cls_loss).mean(dim=0, keepdim=True)
        reg_loss = torch.stack(reg_loss, dim=0) # [B, 12]
        if self.cfg.detector.loss.iou_type != "baseline": iou_loss = torch.stack(iou_loss, dim=0)
        
        # print(f"Befroe dep_loss.requires_grad = {dep_loss[0].requires_grad}")
        if dep_preds != None: dep_loss = torch.stack(dep_loss, dim=0)
        
        # Weight regression loss by number of ground true in each images
        pos_weight = reg_pred.new(n_pos_list).unsqueeze(1) #[B, 1]
        reg_loss_weighted = torch.sum(reg_loss * pos_weight / (torch.sum(pos_weight) + 1e-6), dim=0)
        reg_loss          = reg_loss_weighted.mean(dim=0, keepdim=True)
        
        # Weight IoU loss by number of ground true in each images
        if self.cfg.detector.loss.iou_type != "baseline":
            iou_loss_weighted = torch.sum(iou_loss * pos_weight / (torch.sum(pos_weight) + 1e-6), dim=0)
            iou_loss          = iou_loss_weighted.mean(dim=0, keepdim=True)

        # Weight Depth loss by number of ground true in each images
        if dep_preds != None:
            dep_loss = torch.sum(pos_weight * dep_loss / (torch.sum(pos_weight) + 1e-6), dim=0).mean(dim=0, keepdim=True)
        
        # TODO, re-implement iou_loss someday, also 2d bbox regression can be eliminated if we use iou loss
        log_dict = {}
        log_dict['1/cls_loss']   = cls_loss
        log_dict['1/reg_loss']   = reg_loss
        log_dict['1/total_loss'] = cls_loss + reg_loss
        if self.cfg.detector.depth_branch.is_seperate_cz:
            log_dict['1/dep_loss']    = dep_loss
            log_dict['1/total_loss'] += dep_loss
        if self.cfg.detector.loss.iou_type != "baseline":
            log_dict['1/iou_loss']    = iou_loss
            log_dict['1/total_loss'] += iou_loss
            reg_loss_weighted = torch.cat((reg_loss_weighted.new_zeros(4), reg_loss_weighted), dim=0)
        log_dict['2/dx']     = reg_loss_weighted[0]
        log_dict['2/dy']     = reg_loss_weighted[1]
        log_dict['2/dw']     = reg_loss_weighted[2]
        log_dict['2/dh']     = reg_loss_weighted[3]
        log_dict['2/cdx']    = reg_loss_weighted[4]
        log_dict['2/cdy']    = reg_loss_weighted[5]
        if self.cfg.detector.depth_branch.is_seperate_cz:
            log_dict['2/cdz']    = dep_loss.detach().cpu().numpy()[0]
            log_dict['4/d_sin']  = reg_loss_weighted[6]
            log_dict['4/d_cos']  = reg_loss_weighted[7]
            log_dict['4/dw']     = reg_loss_weighted[8]
            log_dict['4/dh']     = reg_loss_weighted[9]
            log_dict['4/dl']     = reg_loss_weighted[10]
            log_dict['4/d_hdg']  = reg_loss_weighted[11]
        else:
            log_dict['2/cdz']    = reg_loss_weighted[6]
            log_dict['4/d_sin']  = reg_loss_weighted[7]
            log_dict['4/d_cos']  = reg_loss_weighted[8]
            log_dict['4/dw']     = reg_loss_weighted[9]
            log_dict['4/dh']     = reg_loss_weighted[10]
            log_dict['4/dl']     = reg_loss_weighted[11]
            log_dict['4/d_hdg']  = reg_loss_weighted[12]
        
        log_dict['3/n_positive_anchor'] = torch.tensor([n_pos])
        log_dict['3/n_negative_anchor'] = torch.tensor([n_neg])
        # log_dict['3/n_ignored_anchor']  = torch.tensor([n_ign])
        log_dict['3/n_positive_predict'] = torch.numel(cls_pred[ cls_pred >  0 ])
        log_dict['3/n_negative_predict'] = torch.numel(cls_pred[ cls_pred <= 0 ])
        log_dict['3/mean_positive_predict'] = cls_pred[ cls_pred >  0 ].mean()
        log_dict['3/mean_negative_predict'] = cls_pred[ cls_pred <= 0 ].mean()

        return log_dict
