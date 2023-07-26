# Perspective-Aware Convolution
import torch
import torchvision.ops
from torch import nn
import numpy as np
import os
from math import sqrt
import sys
import torch.nn.functional as F
from math import atan2, pi, sin, cos

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
                 offset_2d=(32, 32),
                 input_shape=(18,80),
                 pad_mode="constant",
                 adpative_P2=False,
                 lock_theta_ortho=False,):

        super(PerspectiveConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.mode = mode
        self.offset_2d = offset_2d
        self.h, self.w = input_shape
        self.D_RATIO = 16 # Downsamle ratio
        self.pad_size = 6
        self.pad_mode = pad_mode
        self.adpative_P2 = adpative_P2
        
        self.lock_theta_ortho = lock_theta_ortho
        print(f"self.pad_mode = {self.pad_mode}")
        print(f"self.offset_2d = {self.offset_2d}")
        print(f"self.adpative_P2 = {self.adpative_P2}")
        print(f"self.lock_theta_ortho = {self.lock_theta_ortho}")
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

        self.offset_cache = {}
        
        # For Fix P2
        self.P2_A = kitti_calib_file_parser(os.path.join(CAR_DIR, f"000169.txt"),
                                            new_shape_tf = (288, 1280), 
                                            crop_tf = 100)
        self.offsets_P2_A = self.get_offset(self.P2_A)
        
        #
        # print(f"self.offsets.max() = {self.offsets.max()}") #  4.9375
        # print(f"self.offsets.min() = {self.offsets.min()}") # -4.1875

    def get_offset(self, P2):
        # Use cache to speed up
        if self.adpative_P2 and str(P2) in self.offset_cache: return self.offset_cache[str(P2)]
        
        # offset = np.zeros((8, 18, self.h+self.pad_size*2, self.w+self.pad_size*2))
        offset = np.zeros((1, 18, self.h, self.w))
        
        for v_f_idx in range(self.h):
            for u_f_idx in range(self.w):
                v, u = v_f_idx*self.D_RATIO, u_f_idx*self.D_RATIO
                # 
                slope = get_slope((u, v, AVG_Y3D_CENTER), P2)
                
                # Use Fix 2d offset
                dcx, dcy = self.offset_2d
                if self.lock_theta_ortho: theta = pi/2
                else:                     theta = atan2(slope, 1)
                
                dx, dy = (dcy*cos(theta), dcy*sin(theta))
                for i, i_o in enumerate([-dy, -dcx-dx,   -dy, -dx,   -dy, dcx-dx,
                                           0, -dcx   ,     0,   0,     0, dcx   ,
                                          dy, -dcx+dx,    dy,  dx,    dy, dcx+dx]):
                    offset[:, i, v_f_idx, u_f_idx] = i_o/self.D_RATIO
                
                # dcx = self.offset_2d
                # dcy = self.offset_2d
                # dsx = dcy - sqrt(1 + slope**2)
                # for i, i_o in enumerate( [-dsx*slope, -dcx-dsx,   -dsx*slope, -dsx,   -dsx*slope, dcx-dsx,
                #                            0        , -dcx    ,    0        , 0   ,    0,         dcx    ,
                #                            dsx*slope, -dcx+dsx,    dsx*slope,  dsx,    dsx*slope, dcx+dsx ] ):
                #     offset[:, i, v_f_idx, u_f_idx] = i_o/self.D_RATIO
                
                # offset from regular convolution
                offset[:, :, v_f_idx, u_f_idx] -= np.array([-1,-1,  -1,0,  -1,1,
                                                             0,-1,   0,0,   0,1,
                                                             1,-1,   1,0,   1,1])
        offset = torch.from_numpy(offset).type(torch.float32).cuda()
        
        # Update Cache
        if self.adpative_P2:
            self.offset_cache[str(P2)] = torch.clone(offset)
            print(f"Update offset cache")
            # print(P2)
            # print(offset)
        
        return offset
    
    def forward(self, inputs):
        x  = inputs['features']
        P2 = inputs['P2']
        B = x.shape[0]
        
        # Get offset
        if self.adpative_P2:
            offsets = torch.cat([ self.get_offset(P2[i].cpu().numpy()) for i in range(B) ], dim=0) # [8, 18, 18, 80]
        else:
            offsets = torch.cat([ self.get_offset(self.P2_A)           for _ in range(B) ], dim=0) # [8, 18, 18, 80]
                
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offsets,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=None,
                                          stride=self.stride,)
        return x

class PerspectiveConv2d_cubic(PerspectiveConv2d):
    def get_offset(self, P2):
        # Use cache to speed up
        if self.adpative_P2 and str(P2) in self.offset_cache: return self.offset_cache[str(P2)]
        
        cu, cv = (P2[0, 2], P2[1, 2])
        offset = np.zeros((1, 18, self.h, self.w))
        
        for v_f_idx in range(self.h):
            for u_f_idx in range(self.w):
                v, u = v_f_idx*self.D_RATIO, u_f_idx*self.D_RATIO

                # Cubic offset kernal
                dcx = self.offset_2d
                dcy = self.offset_2d
                theta = atan2(cv - v, cu - u) - pi/2
                
                for i, o in enumerate( [[-dcy,-dcx],  [-dcy,0],  [-dcy,dcx],
                                        [   0,-dcx],  [   0,0],  [   0,dcx],
                                        [ dcy,-dcx],  [ dcy,0],  [ dcy,dcx] ] ):
                    y, x = o[0], o[1]
                    offset[:, 2*i  , v_f_idx, u_f_idx] = (x*sin(theta) + y*cos(theta)) / self.D_RATIO
                    offset[:, 2*i+1, v_f_idx, u_f_idx] = (x*cos(theta) - y*sin(theta)) / self.D_RATIO

                # offset from regular convolution
                offset[:, :, v_f_idx, u_f_idx] -= np.array([-1,-1,  -1,0,  -1,1,
                                                             0,-1,   0,0,   0,1,
                                                             1,-1,   1,0,   1,1])
        offset = torch.from_numpy(offset).type(torch.float32).cuda()
        
        # Update Cache
        if self.adpative_P2:
            self.offset_cache[str(P2)] = torch.clone(offset)
            print(f"Update offset cache")
        
        return offset

