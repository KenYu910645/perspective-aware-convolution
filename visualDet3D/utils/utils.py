import torch
import numpy as np
import cv2
import sys
import os
import tempfile
import shutil
import importlib
from easydict import EasyDict

class LossLogger():
    def __init__(self, recorder, data_split='train'):
        self.recorder = recorder
        self.data_split = data_split
        self.reset()

    def reset(self):
        self.loss_stats = {} # each will be 
    
    def update(self, loss_dict):
        for key in loss_dict:
            if key not in self.loss_stats:
                self.loss_stats[key] = AverageMeter()
            try:
                self.loss_stats[key].update(loss_dict[key].mean().item())
            except:
                self.loss_stats[key].update(loss_dict[key])
    
    def log(self, step):
        for key in self.loss_stats:
            # name = key + '/' + self.data_split
            # name = self.data_split + '/' + key 
            self.recorder.add_scalar(key, self.loss_stats[key].avg, step)

def convertAlpha2Rot(alpha, cx, P2):
    cx_p2 = P2[..., 0, 2]
    fx_p2 = P2[..., 0, 0]
    ry3d = alpha + np.arctan2(cx - cx_p2, fx_p2)
    ry3d[np.where(ry3d > np.pi)] -= 2 * np.pi
    ry3d[np.where(ry3d <= -np.pi)] += 2 * np.pi
    return ry3d

def convertRot2Alpha(ry3d, cx, P2):
    cx_p2 = P2[..., 0, 2]
    fx_p2 = P2[..., 0, 0]
    alpha = ry3d - np.arctan2(cx - cx_p2, fx_p2)
    alpha[alpha > np.pi] -= 2 * np.pi
    alpha[alpha <= -np.pi] += 2 * np.pi
    return alpha

def alpha2theta_3d(alpha, x, z, P2):
    """ Convert alpha to theta with 3D position
    Args:
        alpha [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        theta []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(alpha, torch.Tensor):
        theta = alpha + torch.atan2(x + offset, z)
    else:
        theta = alpha + np.arctan2(x + offset, z)
    return theta

def theta2alpha_3d(theta, x, z, P2):
    """ Convert theta to alpha with 3D position
    Args:
        theta [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        alpha []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(theta, torch.Tensor):
        alpha = theta - torch.atan2(x + offset, z)
    else:
        alpha = theta - np.arctan2(x + offset, z)
    return alpha

def draw_3D_box(img, corners, color = (255, 255, 0)):
    """
        draw 3D box in image with OpenCV,
        the order of the corners should be the same with BBox3dProjector
    """
    points = np.array(corners[0:2], dtype=np.int32) #[2, 8]
    points = [tuple(points[:,i]) for i in range(8)]
    for i in range(1, 5):
        cv2.line(img, points[i], points[(i%4+1)], color, 2)
        cv2.line(img, points[(i + 4)%8], points[((i)%4 + 5)%8], color, 2)
    cv2.line(img, points[2], points[7], color)
    cv2.line(img, points[3], points[6], color)
    cv2.line(img, points[4], points[5],color)
    cv2.line(img, points[0], points[1], color)
    return img

def compound_annotation(labels, max_length, bbox2d, bbox3d, loc_3d_ry, obj_types, calibs):
    """ Compound numpy-like annotation formats. Borrow from Retina-Net
    Args:
        labels: List[List[str]], [B][num_gts] - 'Car'
        max_length: int, max_num_objects, can be dynamic for each iterations
        bbox2d: List[np.ndArray], [left, top, right, bottom]. 
        bbox3d: List[np.ndArray], [cam_x, cam_y, z, w, h, l, alpha].
        loc_3d_ry: [x3d, y3d, z3d, rot_y]
        obj_types: List[str]
        calibs : [B, 3, 4] # [8, 3, 4]
    Return:
    annotations
        np.ndArray, [batch_size, max_length, 16] # (8, 12, 16)
            [x1, y1, x2, y2, cls_index, cx, cy, z, w, h, l, alpha, x3d, y3d, z3d, rot_y]
             0   1   2   3   4          5   6   7  8  9  10 11     12   13   14   15
            cls_index = -1 if empty
    """
    # B = len(labels) # batch size
    
    annotations = np.ones([len(labels), max_length, 16]) * -1
    for i_batch in range(len(labels)):
        label = labels[i_batch]
        for i_gt in range(len(label)): # Number of Groundtrue
            annotations[i_batch, i_gt] = np.concatenate([
                bbox2d[i_batch][i_gt], 
                [obj_types.index(label[i_gt])],
                bbox3d[i_batch][i_gt],
                loc_3d_ry[i_batch][i_gt]])
    

    # The block that works
    # annotations = np.ones([len(labels), max_length, bbox3d[0].shape[-1] + 9]) * -1
    # for i in range(len(labels)):
    #     label = labels[i]
    #     for j in range(len(label)):
    #         annotations[i, j] = np.concatenate([
    #             bbox2d[i][j], [obj_types.index(label[j])], bbox3d[i][j], loc_3d_ry[i][j]
    #         ])
    
    # # print the thread ID for each batch
    # thread_id = threading.get_ident()
    # print(f"Thread ID = {thread_id}")

    # print(annotations.shape) # (8, 12, 16/24)
    return annotations

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cfg_from_file(cfg_filename:str)->EasyDict:
    assert cfg_filename.endswith('.py')
    # print(cfg_filename) # config/data_augumentation/add_cutout_2_hole.py
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py')
        temp_config_name = os.path.basename(temp_config_file.name)
        shutil.copyfile(cfg_filename, os.path.join(temp_config_dir, temp_config_name))
        temp_module_name = os.path.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        cfg = getattr(importlib.import_module(temp_module_name), 'cfg')
        assert isinstance(cfg, EasyDict)
        sys.path.pop(0)
        del sys.modules[temp_module_name]
        temp_config_file.close()

    return cfg

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"Create directory at {path}")