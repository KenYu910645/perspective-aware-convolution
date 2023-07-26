# Perspective-Aware Convolution
import torch
import torchvision.ops
from torch import nn
import numpy as np
import os
from math import sqrt
import sys
import torch.nn.functional as F

sys.path.insert(0, "/home/lab530/KenYu/ml_toolkit/kitti/")
from util_kitti import kitti_calib_file_parser

# Image files 
IMG_DIR = "/home/lab530/KenYu/kitti/training/image_2/"
# Anotations files 
ANO_DIR = "/home/lab530/KenYu/kitti/training/label_2/"
# Calibration files
CAR_DIR = "/home/lab530/KenYu/kitti/training/calib/"

AVG_Y3D_CENTER = 0.94677

def xyz_2_uv(X, P2):
    '''
    Transform camera cooridiante(x,y,z) to image plane(u,v)
    '''
    x,y,z = X
    X_img = np.matmul(P2, np.array([[x], [y], [z], [1]]))
    X_img /= X_img[2]
    u, v = int(X_img[0]), int(X_img[1])
    return (u,v)

def uvy_2_xyz(uvy, P2): # y0=AVG_Y3D_CENTER): # 1.65
    u, v, y0 = uvy
    
    P2_3x3 = np.array([P2[0, :3], P2[1, :3], P2[2, :3]])
    P2_inv = np.linalg.inv(P2_3x3)
    
    cz = y0/( P2_inv[1,0]*u + P2_inv[1,1]*v + P2_inv[1,2] )
    
    ans = np.matmul(P2_inv, np.array([ [u*cz], [v*cz], [cz]]))
    return (ans[0][0], ans[1][0], ans[2][0])

def get_slope(uvy, P2):
    MAX_SLOPE = 500
    
    x, y, z = uvy_2_xyz(uvy, P2)
    u1, v1 = xyz_2_uv((x, y, z-10), P2)
    u2, v2 = xyz_2_uv((x, y, z+10), P2)
    
    # Avoid ZeroDivision
    if (u2-u1) == 0 : return MAX_SLOPE
    
    slope = (v2 - v1) / (u2 - u1)
    
    if slope > MAX_SLOPE:
        return MAX_SLOPE
    elif slope < -MAX_SLOPE:
        return -MAX_SLOPE
    else:
        return slope

class PerspectiveConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mode,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 offset_3d=0.4,
                 input_shape=(18,80),
                 pad_mode="constant"):

        super(PerspectiveConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.mode = mode
        self.offset_3d = offset_3d
        self.h, self.w = input_shape
        self.D_RATIO = 16 # Downsamle ratio
        self.pad_size = 6
        self.pad_mode = pad_mode
        # self.offset_cache = {}
        print(f"self.pad_mode = {self.pad_mode}")
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
    
        self.P2_A = kitti_calib_file_parser(os.path.join(CAR_DIR, f"000169.txt"),
                                            new_shape_tf = (288, 1280), 
                                            crop_tf = 100)
        self.offsets = self.get_offset(self.P2_A)
        
        #
        # print(f"self.offsets.max() = {self.offsets.max()}") #  4.9375
        # print(f"self.offsets.min() = {self.offsets.min()}") # -4.1875

    def get_offset(self, P2):
        # Use cache to speed up
        # if str(P2) in self.offset_cache: return self.offset_cache[str(P2)]
        
        # offset = np.zeros((8, 18, self.h+self.pad_size*2, self.w+self.pad_size*2))
        offset = np.zeros((8, 18, self.h, self.w))
        
        for v_f_idx in range(self.h):
            for u_f_idx in range(self.w):
                v, u = v_f_idx*self.D_RATIO, u_f_idx*self.D_RATIO
                
                dx = self.offset_3d
                dy = self.offset_3d
                x, y, z = uvy_2_xyz((u, v, AVG_Y3D_CENTER), P2 )
                for i, (xi, yi, zi) in enumerate([(x-dx, y-dy, z), (x, y-dy, z), (x+dx, y-dy, z),
                                                  (x-dx, y   , z), (x, y   , z), (x+dx, y   , z),
                                                  (x-dx, y+dy, z), (x, y+dy, z), (x+dx, y+dy, z)]):
                    ui, vi = xyz_2_uv((xi, yi, zi), P2 )
                    # offset[:, i*2  , v_f_idx+self.pad_size, u_f_idx+self.pad_size] = (vi-v)/self.D_RATIO
                    # offset[:, i*2+1, v_f_idx+self.pad_size, u_f_idx+self.pad_size] = (ui-u)/self.D_RATIO
                    offset[:, i*2  , v_f_idx, u_f_idx] = (vi-v)/self.D_RATIO
                    offset[:, i*2+1, v_f_idx, u_f_idx] = (ui-u)/self.D_RATIO

                # offset from regular convolution
                offset[:, :, v_f_idx, u_f_idx] -= np.array([-1,-1,  -1,0,  -1,1,
                                                             0,-1,   0,0,   0,1,
                                                             1,-1,   1,0,   1,1])
                
                # offset[:, :, v_f_idx+self.pad_size, u_f_idx+self.pad_size] -= np.array([-1,-1,  -1,0,  -1,1,
                #                                                                          0,-1,   0,0,   0,1,
                #                                                                          1,-1,   1,0,   1,1])
        offset = torch.from_numpy(offset).type(torch.float32).cuda()
        
        # Update Cache
        # self.offset_cache[str(P2)] = torch.clone(offset)
        # print(f"Update offset cache")
        # print(P2)
        # print(offset)
        
        return offset
    
    def forward(self, inputs):
        x  = inputs['features']
        P2 = inputs['P2']
        B = x.shape[0]
        
        # Get offset
        # offsets = []
        # for i in range(B):
        #     offsets.append( self.get_offset(P2[i]) )
        # offsets = torch.cat(offsets, dim=0)
        
        # Replicate Padding
        # x = F.pad(x, [self.pad_size for _ in range(4)], mode = self.pad_mode) # "replicate"
        # print(f"x = {x.shape}") # [8, 1024, 30, 92]
        
        # TODO maybe we can use dialtion convolution instead of DCN
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=self.offsets[:B, :, :, :],
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=None,
                                          stride=self.stride,)
        # For padding
        # x = x[:, :, 6:24, 6:86]
        
        return x

