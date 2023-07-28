import cv2
import os 
import random 
import shutil
import copy
import json 
import numpy as np 
from math import sqrt
import argparse

from visualDet3D.utils.iou_3d import get_3d_box
from visualDet3D.utils.util_kitti import KITTI_Object

# TODO 
SEGMT_DIR = "/home/lab530/KenYu/kitti/training/image_bbox_label/"

class CopyPaste_Object(KITTI_Object):
    def __init__(self, str_line, tf_matrix, idx_img = None, idx_line = None):
        super().__init__(str_line, tf_matrix, idx_img, idx_line)
        self.corners_3d = get_3d_box((self.l, self.w, self.h), self.rot_y, (self.x3d, self.y3d, self.z3d))
        
        # Find segment labels
        json_path = SEGMT_DIR + f"{idx_img}_{idx_line}.json"
        #
        self.seg_points = [] # Only use if IS_SEG_GT == true
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                raw_json = json.load(f)
                self.seg_points = raw_json['shapes'][0]["points"]
        
        
        # # Load the image
        # self.img_src = imread
        

    def reprojection(self):
        super().reprojection()
        self.corners_3d = get_3d_box((self.l, self.w, self.h), self.rot_y, (self.x3d, self.y3d, self.z3d))
