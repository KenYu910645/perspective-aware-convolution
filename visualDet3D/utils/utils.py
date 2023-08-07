import torch
import numpy as np
import cv2
import os
import importlib
from easydict import EasyDict
import pprint
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

def cfg_from_file(cfg_path:str)->EasyDict:

    assert cfg_path.endswith('.py')
    
    cfg = getattr(importlib.import_module(cfg_path.split('.')[0].replace('/', '.')), 'cfg')

    cfg.path = EasyDict()
    cfg.path.project_path      = os.path.join('exp_output', cfg_path.split('/')[1], cfg_path.split('/')[2].split('.')[0])
    cfg.path.log_path          = os.path.join(cfg.path.project_path, "log")
    cfg.path.checkpoint_path   = os.path.join(cfg.path.project_path, "checkpoint")
    cfg.path.preprocessed_path = os.path.join(cfg.path.project_path, "output")
    cfg.path.train_imdb_path   = os.path.join(cfg.path.project_path, "output", "training")
    cfg.path.train_disp_path   = os.path.join(cfg.path.project_path, "output", "training", "disp")
    cfg.path.val_imdb_path     = os.path.join(cfg.path.project_path, "output", "validation")
    cfg.path.test_imdb_path    = os.path.join(cfg.path.project_path, "output", "testing")

    create_dir(cfg.path.project_path)
    create_dir(cfg.path.log_path)
    create_dir(cfg.path.checkpoint_path)
    create_dir(cfg.path.preprocessed_path)
    create_dir(cfg.path.train_imdb_path)
    create_dir(cfg.path.val_imdb_path)
    create_dir(cfg.path.train_disp_path)
    create_dir(cfg.path.test_imdb_path)

    return cfg

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"Create directory at {path}")