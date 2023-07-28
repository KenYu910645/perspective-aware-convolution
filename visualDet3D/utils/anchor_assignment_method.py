import torch
from visualDet3D.utils.util_kitti import calc_iou

def maxIoU(anchors_tensor, labels_tensor):
    '''
    Anchor assignment via Max IOU
    '''
    BG_IOU_THRES = 0.4
    FG_IOU_THRES = 0.5

    # Convert label to tensor
    IoU = calc_iou(anchors_tensor, labels_tensor) # IoU = [14284, 1]
    # Find max overlap groundture with the anchor
    iou_max, iou_argmax = IoU.max(dim=1) # [14284], [14284]
    # Anchor Assignment
    anchor_assignment = iou_argmax.new_full((iou_argmax.shape[0], ), -2, dtype=torch.long)
    # Assign positive anchor
    anchor_assignment[iou_max >= FG_IOU_THRES] = iou_argmax[iou_max >= FG_IOU_THRES]
    # Assign negative anchor
    anchor_assignment[iou_max <  BG_IOU_THRES] = -1
    # Get index of postive and negative anchor
    pos_inds = torch.nonzero(anchor_assignment >= 0, as_tuple=False).squeeze(dim=1)
    neg_inds = torch.nonzero(anchor_assignment ==-1, as_tuple=False).squeeze(dim=1)

    return pos_inds, neg_inds, anchor_assignment