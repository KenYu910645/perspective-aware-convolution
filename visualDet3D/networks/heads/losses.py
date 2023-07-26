from easydict import EasyDict
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from visualDet3D.networks.utils.utils import calc_iou
from visualDet3D.networks.lib.disparity_loss import stereo_focal_loss
from visualDet3D.utils.timer import profile
from torch.nn.functional import logsigmoid

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, balance_weights=torch.tensor([1.0], dtype=torch.float)):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.register_buffer("balance_weights", balance_weights)

    def forward(self, cls_pred:torch.Tensor, 
                      targets:torch.Tensor, 
                      gamma:Optional[float]=None, 
                      balance_weights:Optional[torch.Tensor]=None)->torch.Tensor:
        """
            input:
                cls_pred  :[..., num_classes]  linear output
                targets         :[..., num_classes] == -1(ignored), 0, 1
            return:
                cls_loss        :[..., num_classes]  loss with 0 in trimmed or ignored indexes 
            balance_weights: Emphasis on positive sample, it's 20 times more important than background
            gamma == 2.0 in config.py
        """
        if gamma is None:
            gamma = self.gamma
        if balance_weights is None:
            balance_weights = self.balance_weights

        probs = torch.sigmoid(cls_pred) #[B, N, 1]
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
        focal_weight = torch.pow(focal_weight, gamma)

        # Binary Cross1
        bce = -(     targets  * logsigmoid( cls_pred)) * balance_weights +\
              -((1 - targets) * logsigmoid(-cls_pred)) #[B, N, 1]
        
        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors, ignore -1 entry
        # print( torch.numel( targets[ targets == -1.0 ] ) )  # 0
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) #0.02**2 * log(0.98) = 8e-6

        return cls_loss

class SoftmaxFocalLoss(nn.Module):
    def forward(self, classification:torch.Tensor, 
                      targets:torch.Tensor, 
                      gamma:float, 
                      balance_weights:torch.Tensor)->torch.Tensor:
        ## Calculate focal loss weights
        probs = torch.softmax(classification, dim=-1)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
        focal_weight = torch.pow(focal_weight, gamma)

        bce = -(targets * torch.log_softmax(classification, dim=1))

        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) #0.02**2 * log(0.98) = 8e-6
        cls_loss = cls_loss * balance_weights
        return cls_loss

class ModifiedSmoothL1Loss(nn.Module):
    def __init__(self, L1_regression_alpha:float):
        super(ModifiedSmoothL1Loss, self).__init__()
        self.alpha = L1_regression_alpha

    def forward(self, normed_targets:torch.Tensor, pos_reg:torch.Tensor):
        regression_diff = torch.abs(normed_targets - pos_reg) #[K, 12]
        ## Smoothed-L1 formula:
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / self.alpha),
            0.5 * self.alpha * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / self.alpha
        )
        ## clipped to avoid overfitting
        regression_loss = torch.where(
           torch.le(regression_diff, 0.01),
           torch.zeros_like(regression_loss),
           regression_loss
        )

        return regression_loss

class IoULoss(nn.Module):
    """Some Information about IoULoss"""
    def forward(self, preds:torch.Tensor, targs:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        """IoU Loss
        Args:
            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
            targs (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]
        Returns:
            torch.Tensor: [-log(iou)] [*]
        """
        # overlap
        lt = torch.max(preds[..., :2], targs[..., :2]) # Most left-top point of intersection
        rb = torch.min(preds[..., 2:], targs[..., 2:]) # Most right-bottom point of intersection
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        # union
        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1]) # Area perdict
        ag = (targs[..., 2] - targs[..., 0]) * (targs[..., 3] - targs[..., 1]) # Area groundtrue
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union
        ious = torch.clamp(ious, min=eps)

        return -ious.log()

class DIoULoss(nn.Module):
    """Some Information about IoULoss"""
    def forward(self, preds:torch.Tensor, targs:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        """DIoU Loss
        Args:
            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
            targs (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]
        Returns:
            torch.Tensor: [-log(iou)] [*]
        """
        # overlap
        lt = torch.max(preds[..., :2], targs[..., :2]) # Most left-top point of intersection
        rb = torch.min(preds[..., 2:], targs[..., 2:]) # Most right-bottom point of intersection
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        # union
        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1]) # Area perdict
        ag = (targs[..., 2] - targs[..., 0]) * (targs[..., 3] - targs[..., 1]) # Area groundtrue
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union
        ious = torch.clamp(ious, min=eps)

        # Get distance between center d_2
        xc_pred = (preds[..., 2] + preds[..., 0]) / 2
        yc_pred = (preds[..., 3] + preds[..., 1]) / 2
        xc_targ = (targs[..., 2] + targs[..., 0]) / 2
        yc_targ = (targs[..., 3] + targs[..., 1]) / 2
        d_2 = torch.pow(xc_targ - xc_pred, 2) + torch.pow(yc_targ - yc_pred, 2)

        # Get diagnal distance
        lt = torch.min(preds[..., :2], targs[..., :2]) # Most left-top point of two bbox
        rb = torch.max(preds[..., 2:], targs[..., 2:]) # Most right-bottom point of tow bbox
        c_2 = torch.pow(lt[..., 0] - rb[..., 0], 2) + torch.pow(lt[..., 1] - rb[..., 1], 2)
        
        return 1 - ious + d_2/c_2

class CIoULoss(nn.Module):
    """Some Information about IoULoss"""
    def forward(self, preds:torch.Tensor, targs:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        """CIoU Loss
        Args:
            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
            targs (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]
        Returns:
            torch.Tensor: [-log(iou)] [*]
        """
        # overlap
        lt = torch.max(preds[..., :2], targs[..., :2]) # Most left-top point of intersection
        rb = torch.min(preds[..., 2:], targs[..., 2:]) # Most right-bottom point of intersection
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        # union
        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1]) # Area perdict
        ag = (targs[..., 2] - targs[..., 0]) * (targs[..., 3] - targs[..., 1]) # Area groundtrue
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union
        ious = torch.clamp(ious, min=eps)

        # Get distance between center d_2
        xc_pred = (preds[..., 2] + preds[..., 0]) / 2
        yc_pred = (preds[..., 3] + preds[..., 1]) / 2
        xc_targ = (targs[..., 2] + targs[..., 0]) / 2
        yc_targ = (targs[..., 3] + targs[..., 1]) / 2
        d_2 = torch.pow(xc_targ - xc_pred, 2) + torch.pow(yc_targ - yc_pred, 2)

        # Get diagnal distance c_2
        lt = torch.min(preds[..., :2], targs[..., :2]) # Most left-top point of two bbox
        rb = torch.max(preds[..., 2:], targs[..., 2:]) # Most right-bottom point of tow bbox
        c_2 = torch.pow(lt[..., 0] - rb[..., 0], 2) + torch.pow(lt[..., 1] - rb[..., 1], 2)
        
        # aspect ratio V
        w_pred = preds[..., 2] - preds[..., 0]
        h_pred = preds[..., 3] - preds[..., 1]
        w_targ = targs[..., 2] - targs[..., 0]
        h_targ = targs[..., 3] - targs[..., 1]
        V = 4 / np.pi**2 * torch.pow( torch.atan2(w_pred, h_pred) - torch.atan2(w_targ, h_targ) , 2)
        
        # Get weight matrix alpha
        alpha = V / ((1-ious) + V)
        alpha[ious < 0.5] = 0 # alpha == 0 if IOU < 0.5

        return 1 - ious + d_2/c_2 + alpha*V

class DisparityLoss(nn.Module):
    """Some Information about DisparityLoss"""
    def __init__(self, maxdisp:int=64):
        super(DisparityLoss, self).__init__()
        #self.register_buffer("disp",torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])))
        self.criterion = stereo_focal_loss.StereoFocalLoss(maxdisp)

    def forward(self, x:torch.Tensor, label:torch.Tensor)->torch.Tensor:
        #x = torch.softmax(x, dim=1)
        label = label.cuda().unsqueeze(1)
        loss = self.criterion(x, label, variance=0.5)
        #mask = (label > 0) * (label < 64)
        #loss = nn.functional.smooth_l1_loss(disp[mask], label[mask])
        return loss
