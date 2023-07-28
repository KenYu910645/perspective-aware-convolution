from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib

import torch
import numpy as np 
from math import pi, atan2
import pickle
import copy
import cv2

P2_dict = {
            '[[7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01]\n ' +
             '[0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01]\n ' +
             '[0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03]]':'A',
            '[[ 7.070493e+02  0.000000e+00  6.040814e+02  4.575831e+01]\n ' +
             '[ 0.000000e+00  7.070493e+02  1.805066e+02 -3.454157e-01]\n ' +
             '[ 0.000000e+00  0.000000e+00  1.000000e+00  4.981016e-03]]':'B',
            '[[ 7.183351e+02  0.000000e+00  6.003891e+02  4.450382e+01]\n ' + 
             '[ 0.000000e+00  7.183351e+02  1.815122e+02 -5.951107e-01]\n ' + 
             '[ 0.000000e+00  0.000000e+00  1.000000e+00  2.616315e-03]]':'C',
            '[[ 7.188560e+02  0.000000e+00  6.071928e+02  4.538225e+01]\n ' +
             '[ 0.000000e+00  7.188560e+02  1.852157e+02 -1.130887e-01]\n ' +
             '[ 0.000000e+00  0.000000e+00  1.000000e+00  3.779761e-03]]':'D'}

shape_dict_inv  = {(375, 1242, 3): 'A',
                   (370, 1224, 3): 'B',
                   (374, 1238, 3): 'C',
                   (376, 1241, 3): 'D'}
shape_dict = {'A': (375, 1242, 3),
              'B': (370, 1224, 3),
              'C': (374, 1238, 3),
              'D': (376, 1241, 3)}

def kitti_predi_file_parser(predi_file_path, tf_matrix):
    with open(predi_file_path) as f:
        lines = f.read().splitlines()
        lines = list(lines for lines in lines if lines) # Delete empty lines
    return [KITTI_Object(str_line,
                         idx_img = predi_file_path.split('/')[-1].split('.')[0],
                         idx_line = idx_line, 
                         tf_matrix = tf_matrix) for idx_line, str_line in enumerate(lines)]

def kitti_label_file_parser(label_file_path, tf_matrix):
    with open(label_file_path) as f:
        lines = f.read().splitlines()
        lines = list(lines for lines in lines if lines) # Delete empty lines
    return [KITTI_Object(str_line + " NA",
                         idx_img = label_file_path.split('/')[-1].split('.')[0],
                         idx_line = idx_line, 
                         tf_matrix = tf_matrix) for idx_line, str_line in enumerate(lines)]

def kitti_calib_file_parser(calib_file_path, new_shape_tf = None, crop_tf = 0):
    '''
    new_shape_tf = (img_new_h, img_new_w)
    '''
    with open(calib_file_path) as f:
        lines = f.read().splitlines()
        for line in lines:
            if 'P2:' in line.split():
                P2 = np.array([float(i) for i in line.split()[1:]] )
                P2 = np.reshape(P2, (3,4))
                
                P2_type = P2_dict[ str(P2) ]
                
                # Crop Top Transformation
                P2[1, 2] -= crop_tf            # cy' = cy - dv
                P2[1, 3] -= crop_tf * P2[2, 3] # ty' = ty - dv * tz
                
                # Resize Transformation , Perserve aspect ratio
                if new_shape_tf != None:
                    P2[0, :] *= new_shape_tf[0] / (shape_dict[P2_type][0] - crop_tf)
                    P2[1, :] *= new_shape_tf[0] / (shape_dict[P2_type][0] - crop_tf)
                
                return P2

def gac_original_anchor_parser(pkl_path, tf_matrix):
    '''
    pkl_path = '/home/lab530/KenYu/visualDet3D/anchor/max_occlusion_2_anchor.pkl'
    '''
    # Load GAC's anchor
    with open(pkl_path, 'rb') as f:
        anchors = pickle.load(f)

    # anchors['anchors'] = [x1, y1, x2, y2]
    # print(f"anchors['anchors'] = {anchors['anchors'].shape}") # [1, 46080, 4]
    # print(f"anchors['mask'] = {anchors['mask'].shape}") # [8, 46080]
    # # [cz, sin(alpha*2), cos(alpha*2), w, h , l]
    # print(f"anchors['anchor_mean_std_3d'] = {anchors['anchor_mean_std_3d'].shape}") # [46080, 1, 6, 2], z, sin(\t), cos(\t)

    anchor_2D = anchors['anchors'][0, :].detach().cpu().numpy()
    anchor_3D = anchors['anchor_mean_std_3d'][:, 0, :, :].detach().cpu().numpy()
    anchor_mask = anchors['mask'][0, :].detach().cpu().numpy()

    unique, counts = np.unique(anchor_mask, return_counts=True)
    anchor_assign = dict(zip(unique, counts)) # {False: 26824, True: 19256}
    print(f"Useful anchor = {anchor_assign[True]} / {( anchor_assign[True] + anchor_assign[False] )} ")

    anchor_objects = []
    for i in range(anchor_2D.shape[0]):
        # Ignore filtered anchor
        if not anchor_mask[i]: continue

        # convert (cx, cy ,cz) to (x3d, y3d, z3d) # TODO, bug, i don't think 2d center is equal to 3d center
        cx = (anchor_2D[i, 2] + anchor_2D[i, 0]) / 2.0
        cy = (anchor_2D[i, 3] + anchor_2D[i, 1]) / 2.0
        cz =  anchor_3D[i, 0, 0]
        loc_3d = np.linalg.inv(tf_matrix[:, :3]) @ np.array([[cx*cz], [cy*cz], [cz]])
        loc_3d[1, 0] += anchor_3D[i, 4, 0] / 2.0

        # Get observation angle: alpha
        alpha = atan2(anchor_3D[i, 1, 0], anchor_3D[i, 2, 0]) / 2.0
        rot_y = alpha - atan2(loc_3d[2, 0], loc_3d[0, 0]) + pi/2
        # rot_y in [-pi, pi]
        if   rot_y >  pi: rot_y -= 2*pi
        elif rot_y < -pi: rot_y += 2*pi

        # 'category, truncated, occluded alpha, xmin, ymin, xmax, ymax, height, width, length, x3d, y3d, z3d, rot_y, score]
        # 1         2          3        4      5     6     7     8     9       10     11      12   13   14   15     16
        str_line = f"Car NA NA NA {anchor_2D[i, 0]} {anchor_2D[i, 1]} {anchor_2D[i, 2]} {anchor_2D[i, 3]} {anchor_3D[i, 4, 0]} {anchor_3D[i, 3, 0]} {anchor_3D[i, 5, 0]} {loc_3d[0, 0]} {loc_3d[1, 0]} {loc_3d[2, 0]} {rot_y} NA"
        anchor_objects.append(KITTI_Object(str_line, tf_matrix))
    
    return anchor_2D, anchor_3D, anchor_mask, anchor_objects

# 
# # assume crop_top 100 pixel and resize into (288, 1280)
# CROP_TOP = 100
# img_new_h, img_new_w = (288, 1280) # img_new_h, img_new_w

# img = cv2.imread("/home/lab530/KenYu/kitti/training/image_2/000169.png")
# img_ori_h, img_ori_w, _ = img.shape
# img_crp_h = img_ori_h - CROP_TOP

# # All the image has the same P2
# P2 = kitti_calib_file_parser("/home/lab530/KenYu/kitti/training/calib/000169.txt")
# # Transform P2 calibration matrix
# P2_tf = copy.deepcopy(P2)
# # Crop Top 
# P2_tf[1, 2] = P2_tf[1, 2] - CROP_TOP               # cy' = cy - dv
# P2_tf[1, 3] = P2_tf[1, 3] - CROP_TOP * P2_tf[2, 3] # ty' = ty - dv * tz
# # Resize (Preserved aspect ratio)
# P2_tf[0, :] = P2_tf[0, :] * img_new_h / img_crp_h
# P2_tf[1, :] = P2_tf[1, :] * img_new_h / img_crp_h

# P2_fpn_tf = copy.deepcopy(P2) # TODO
# # Resize
# P2_fpn_tf[0, :] *= 384 / img_ori_h
# P2_fpn_tf[1, :] *= 384 / img_ori_h

# Statistic
AVG_HEIGT = 1.526
AVG_WIDTH = 1.629
AVG_LENTH = 3.884

STD_HEIGT = 0.137
STD_WIDTH = 0.102
STD_LENTH = 0.426

ANCHOR_Y_3D_MEAN = 1.71
ANCHOR_Y_3D_STD  = 0.38574

class KITTI_Object:
    def __init__(self, str_line, tf_matrix, idx_img = None, idx_line = None, center_2d = None):
        # str_line == 'Car 0.00 0 -1.58 587.19 178.91 603.38 191.75 1.26 1.60 3.56 -1.53 1.89 73.44 -1.60 1.0'
        #             'category, truncated, occluded alpha, xmin, ymin, xmax, ymax, height, width, length, x3d, y3d, z3d, rot_y, score]
        #              1         2          3        4      5     6     7     8     9       10     11      12   13   14   15     16
        # idx_img = "000123"
        # idx_line is i-th line in the label.txt
        
        # Parse str_line
        sl = str_line.split()
        assert len(sl) == 16, 'KITTI_Object must get 16 argument in str_line, fill in NA if it is not available'
        self.category, self.truncated, self.occluded, self.alpha, self.xmin, self.ymin, self.xmax, self.ymax, self.h, self.w, self.l, self.x3d, self.y3d, self.z3d, self.rot_y, self.score = sl
        # 
        self.raw_str = str_line
        self.idx_img  = idx_img # which image does this obj belong to
        self.idx_line = idx_line # which line does this obj belong to in label.txt
        
        # Get P2
        self.P2 = tf_matrix
        
        # Basic information
        self.h, self.w, self.l = (float(self.h), float(self.w), float(self.l))
        self.x3d, self.y3d, self.z3d = (float(self.x3d), float(self.y3d), float(self.z3d))
        self.rot_y = float(self.rot_y)
        
        # Get corner 2D
        self.corner_2D = get_corner_2D(self.P2, (self.x3d, self.y3d, self.z3d), self.rot_y, (self.l, self.h, self.w))
        
        # Get truncated and occluded
        if self.truncated != "NA": self.truncated = float(self.truncated)
        if self.occluded  != "NA": self.occluded  = int(self.occluded)

        # Get 2d boudning box # TODO transform???
        if self.xmin == "NA": #  or (self.P2 == P2).all():
            # Get 2d boudning box via 3D boudning box when it's not explict assigned
            self.xmin = self.corner_2D[0].min()
            self.ymin = self.corner_2D[1].min()
            self.xmax = self.corner_2D[0].max()
            self.ymax = self.corner_2D[1].max()
        else:
            self.xmin, self.ymin, self.xmax, self.ymax = [int(float(i)) for i in sl[4:8]]
        # Get condident score
        if self.score != "NA": self.score = float(self.score)
        
        # Get observation angle: alpha
        if self.alpha != "NA":
            self.alpha = float(self.alpha)
        else:
            # Get alpha via rot_y if it's not avialable
            self.alpha = self.rot_y + atan2(self.z3d, self.x3d) - pi/2
            if   self.alpha >  pi: self.alpha -= 2*pi # make alpha in [-pi, pi]
            elif self.alpha < -pi: self.alpha += 2*pi

        ##############################
        ### Additional Information ###
        ##############################
        # Get 2D bounding box area
        self.area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        self.y3d_center = self.y3d - self.h/2

        # TODO I don't know why cx,cy,cz ->x3d,y3d,z3d -> cx,cy,cz incur big error
        if center_2d == None:
            # Get cx, cy, cz
            tmp = np.dot(self.P2, np.array([[self.x3d], [self.y3d - self.h/2], [self.z3d], [1]]))
            tmp[0:2] /= tmp[2]
            self.cx, self.cy, self.cz = float(tmp[0]), float(tmp[1]), float(tmp[2])
        else:
            self.cx, self.cy, self.cz = center_2d
        
        # Get cx, cy in 18*80 feature map # I use cx,cy to define anchor's features
        # self.cx_f_index = int(round(self.cx * (1/16))) # 79.5 -> 80 -> IndexError
        # self.cy_f_index = int(round(self.cy * (1/16)))
        
        self.cx_f_index = int(self.cx * (1/16))
        self.cy_f_index = int(self.cy * (1/16))
        # if self.cx_f_index >= 80:
        #     print(self.cx)

    def __str__(self):
        # return self.raw_str
        # Note that this function will output transformed 2D pixels
        return f"{self.category} {self.truncated} {self.occluded} {round(self.alpha, 2)} {round(self.xmin, 2)} {round(self.ymin, 2)} {round(self.xmax, 2)} {round(self.ymax, 2)} {round(self.h, 2)} {round(self.w, 2)} {round(self.l, 2)} {round(self.x3d, 2)} {round(self.y3d, 2)} {round(self.z3d, 2)} {round(self.rot_y, 2)}"

    # def transform_2d_bbox(self, img_ori_h, crop_tf = 0, resize_tf = None):
    def transform_2d_bbox(self):
        '''
        Transform 2D bounding box by P2, this only will be used when groundtrue's 2d box need to transform
        '''
        # Get 2d boudning box via 3D boudning box when it's not explict assigned
        self.xmin = self.corner_2D[0].min()
        self.ymin = self.corner_2D[1].min()
        self.xmax = self.corner_2D[0].max()
        self.ymax = self.corner_2D[1].max()
        return

    def transform_2d_bbox_manually(self, img_ori_h, crop_tf = 0, resize_tf = None):
        # For GAC
        self.ymin -= crop_tf
        self.ymax -= crop_tf
        if resize_tf != None:
            self.xmin *= resize_tf[0] /(img_ori_h - crop_tf)
            self.ymin *= resize_tf[0] /(img_ori_h - crop_tf)
            self.xmax *= resize_tf[0] /(img_ori_h - crop_tf)
            self.ymax *= resize_tf[0] /(img_ori_h - crop_tf)
        
        # This is for pytorch-retinanetfpn, 
        # label.xmin *= 1280/img_ori_w
        # label.ymin *= 384 /img_ori_h
        # label.xmax *= 1280/img_ori_w
        # label.ymax *= 384 /img_ori_h

    def reprojection(self):
        '''
        Use (x3d, y3d, z3d, h, w, l, rot_y) and P2 to reproject everything
        '''
        # Get corner 2D
        self.corner_2D = get_corner_2D(self.P2, (self.x3d, self.y3d, self.z3d), self.rot_y, (self.l, self.h, self.w))
        
        # Get 2dbbox
        self.xmin = int(self.corner_2D[0].min())
        self.ymin = int(self.corner_2D[1].min())
        self.xmax = int(self.corner_2D[0].max())
        self.ymax = int(self.corner_2D[1].max())

        # Get image dimensino
        img_h, img_w, img_c = shape_dict[ P2_dict[ str(self.P2) ] ]

        # # 2D bbox saturation
        # self.xmin = int(max(self.xmin, 0))
        # self.ymin = int(max(self.ymin, 0))
        # self.xmax = int(min(self.xmax, img_w-1))
        # self.ymax = int(min(self.ymax, img_h-1))

        # Get alpha via rot_y if it's not avialable
        self.alpha = self.rot_y + atan2(self.z3d, self.x3d) - pi/2
        if   self.alpha >  pi: self.alpha -= 2*pi # make alpha in [-pi, pi]
        elif self.alpha < -pi: self.alpha += 2*pi

        ##############################
        ### Additional Information ###
        ##############################
        # Get 2D bounding box area
        self.area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        self.y3d_center = self.y3d - self.h/2

        # TODO I don't know why cx,cy,cz ->x3d,y3d,z3d -> cx,cy,cz incur big error
        # Get cx, cy, cz
        tmp = np.dot(self.P2, np.array([[self.x3d], [self.y3d - self.h/2], [self.z3d], [1]]))
        tmp[0:2] /= tmp[2]
        self.cx, self.cy, self.cz = float(tmp[0]), float(tmp[1]), float(tmp[2])

        self.cx_f_index = int(self.cx * (1/16))
        self.cy_f_index = int(self.cy * (1/16))
        
        # Update str-Line
        self.raw_str = self.__str__()

   
def get_corner_2D(P2, loc_3d, rot_y, dimension):
    # Get corner that project to 2D image plane
    x3d, y3d, z3d = loc_3d
    l, h, w = dimension
    
    R = np.array([[ np.cos(rot_y), 0, np.sin(rot_y)],
                  [ 0,             1, 0            ],
                  [-np.sin(rot_y), 0, np.cos(rot_y)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h     for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([x3d, y3d, z3d]).reshape((3, 1))

    # Avoid z_3d < 0, saturate at 0.0001
    corners_3D[2] = np.array([max(0.0001, i) for i in corners_3D[2]])
    # 
    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]
    
    return corners_2D

def draw_birdeyes(ax2, anchor, color, title = "123", is_print_confident = False):
    # Draw GT
    gt_corners_2d = compute_birdviewbox(anchor)
    
    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    # print(Path.LINETO)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color=color, label=title)
    ax2.add_patch(p)
    # Draw conf text
    # if len(line) == 16: # Prediction 
    # if is_print_confident: # TODO need to fix this
    #     conf = round(float(line[-1]), 2)
    #     ax2.text(max(gt_corners_2d[:, 0]), max(gt_corners_2d[:, 1]),
    #             str(conf), fontsize=8, color = (1, 0, 0))

def calc_iou(a, b):
    """
    This is from GAC code
    Calculate 2D iou between two bounding box
    INput: 
        a - [:, 4] - x1, y1, x2, y2
    """

    # Get area of bounding boxes
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # Get Intersection
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih

    # Get Union
    union = torch.unsqueeze(a_area, dim=1) + b_area - intersection
    union = torch.clamp(union, min=1e-8)
    
    # Get IoU
    IoU = intersection / union

    return IoU

def compute_birdviewbox(anchor):
    BEV_SHAPE = 900
    BEV_SCALE = 15
    h = anchor.h * BEV_SCALE
    w = anchor.w * BEV_SCALE
    l = anchor.l * BEV_SCALE
    x = anchor.x3d * BEV_SCALE
    y = anchor.y3d * BEV_SCALE
    z = anchor.z3d * BEV_SCALE
    rot_y = anchor.rot_y

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [ np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [-l/2, l/2, l/2, -l/2]  # -l/2
    z_corners = [ w/2, w/2,-w/2, -w/2]  # -w/2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(BEV_SHAPE/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T
    return np.vstack((corners_2D, corners_2D[0,:]))

def xz2bev(x3d, z3d):
    return (x3d * 15 + int(900/2), 
            z3d * 15)

def draw_corner_2D(ax, corners_2D, color = (1,0,0), is_draw_front = True):
    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO

    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=1)
    ax.add_patch(p)

    # put a mask on the front
    if is_draw_front:
        width  = corners_2D[:, 3][0] - corners_2D[:, 1][0]
        height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
        front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
        ax.add_patch(front_fill)

def draw_2Dbox(ax, corners, color = (1,0,0)):
    x1, y1, x2, y2 = corners
    width  = x2 - x1
    height = y2 - y1
    front_fill = patches.Rectangle((x1, y1),
                                    width, 
                                    height, 
                                    fill=False, 
                                    color=color, 
                                    linewidth=1,
                                    alpha=1)
    ax.add_patch(front_fill)

def set_bev_background(ax):
    BEV_SHAPE = 900
    x1 = np.linspace(0, BEV_SHAPE/2)
    x2 = np.linspace(BEV_SHAPE/2, BEV_SHAPE)
    ax.plot(x1, BEV_SHAPE / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax.plot(x2, x2 - BEV_SHAPE / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax.plot(BEV_SHAPE / 2, 0, marker='+', markersize=16, markeredgecolor='red')
    ax.imshow(np.zeros((BEV_SHAPE, BEV_SHAPE, 3), np.uint8), origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])

def init_img_plt_with_plasma(imgs, titles = None):
    '''
    num_plot: how many images does it have 
    '''
    num_plot = len(imgs)

    fig = plt.figure(figsize=(18, 5*num_plot), dpi=100)
    fig.set_facecolor('white')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)
    
    gs = GridSpec(num_plot, 4)
    gs.update(wspace=0)  # set the spacing between axes

    axs = [] # [ax_img, ......]
    for i in range(num_plot):
        axs.append( fig.add_subplot(gs[i, :]) )

    for i, ax in enumerate(axs):
        # Draw images
        ax.axis('off')

        # cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        
        im = ax.imshow(imgs[i], cmap='plasma') # jet plasma jet_r

        # fig.colorbar(im, cax=cax, orientation='vertical')
        # cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        # cbar.ax.invert_yaxis()


        # Set titles
        if not titles is None:
            ax.set_title(titles[i], fontsize=25)
    
    return axs # [ax_img, ......]

def init_img_plt_without_bev(imgs, titles = None):
    '''
    num_plot: how many images does it have 
    '''
    num_plot = len(imgs)

    fig = plt.figure(figsize=(18, 5*num_plot), dpi=100)
    fig.set_facecolor('white')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)
    
    gs = GridSpec(num_plot, 4)
    gs.update(wspace=0)  # set the spacing between axes

    axs = [] # [ax_img, ......]
    for i in range(num_plot):
        axs.append( fig.add_subplot(gs[i, :]) )

    for i, ax in enumerate(axs):
        # Draw images
        ax.axis('off')
        ax.imshow(imgs[i][...,::-1])
        
        # Set titles
        if not titles is None: ax.set_title(titles[i], fontsize=25)
    
    return axs # [ax_img, ......]

def init_img_plt(imgs, titles = None):
    '''
    num_plot: how many images does it have 
    '''
    num_plot = len(imgs)

    fig = plt.figure(figsize=(18, 5*num_plot), dpi=100)
    fig.set_facecolor('white')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)
    
    gs = GridSpec(num_plot, 4)
    gs.update(wspace=0)  # set the spacing between axes

    axs = [] # [(ax_img, ax_bev), ......]
    for i in range(num_plot):
        axs.append( (fig.add_subplot(gs[i, :3]), fig.add_subplot(gs[i,  3])) )

    for i, ax in enumerate(axs):
        # Draw images        
        ax[0].axis('off')
        ax[0].imshow(imgs[i][...,::-1])
        
        # Set titles
        if not titles is None:
            ax[0].set_title(titles[i], fontsize=25)
        
        # Set background for BEV plot
        set_bev_background(ax[1])
    
    return axs # [(ax_img, ax_bev), ......]

def init_zy_plt(titles):
    num_plot = len(titles)
    fig = plt.figure(figsize=(30, 15*num_plot), dpi=100)
    # plt.subplots_adjust(wspace=0, hspace=0)
    gs = GridSpec(num_plot, 1)
    gs.update(wspace=0)  # set the spacing between axes.
    axs = [fig.add_subplot(gs[i]) for i in range(num_plot)]

    # plot GAC all anchor
    for i, ax in enumerate(axs): 
        ax.set_title(titles[i], fontsize=25)
        ax.set_xlabel("Z", fontsize=25)
        ax.set_ylabel("Y", fontsize=25)
        ax.set_xlim([0, 80])
        ax.set_ylim([-3, 5])
    return axs

def init_xz_plt(titles):
    num_plot = len(titles)
    fig = plt.figure(figsize=(30, 15*num_plot), dpi=100)
    # plt.subplots_adjust(wspace=0, hspace=0)
    gs = GridSpec(num_plot, 1)
    gs.update(wspace=0)  # set the spacing between axes.
    axs = [fig.add_subplot(gs[i]) for i in range(num_plot)]

    # plot GAC all anchor
    for i, ax in enumerate(axs): 
        ax.set_title(titles[i], fontsize=25)
        ax.set_xlabel("X", fontsize=25)
        ax.set_ylabel("Z", fontsize=25)
        ax.set_xlim([-40, 40])
        ax.set_ylim([0, 80])
    return axs

def load_tf_image(img_path):
    # assume crop_top 100 pixel and resize into (288, 1280)
    CROP_TOP = 100
    img_new_h, img_new_w = (288, 1280) # img_new_h, img_new_w

    img = cv2.imread(img_path)
    img_ori_h, img_ori_w, _ = img.shape
    img_crp_h = img_ori_h - CROP_TOP

    # Assuming croptop 100 pixel + resize into (288, 1280) - preservse aspect ratio
    img_tf = cv2.resize(img[CROP_TOP:, :], (int(img_ori_w * img_new_h / img_crp_h), 
                                            int(img_crp_h * img_new_h / img_crp_h)))
    # crop in
    if img_tf.shape[1] > img_new_w:
        img_tf = img_tf[:, 0:img_new_w, :]
    elif img_tf.shape[1] < img_new_w: # pad out
        img_tf = np.pad(img_tf,  [(0, 0), (0, img_new_w - img_tf.shape[1]), (0, 0)], 'constant')
    # print(f"img_tf = {img_tf.shape}") # (288, 1280)

    return img_tf

def load_tf_image_fpn(img_path):
    img = cv2.imread(img_path)
    img_tf = cv2.resize(img, (1280, 384))
    return img_tf
