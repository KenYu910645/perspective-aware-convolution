# Perspective-Aware Convolution
import torch
import torchvision.ops
from torch import nn
import numpy as np
import os
from math import sqrt
import sys

from visualDet3D.utils.util_kitti import kitti_calib_file_parser

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
                 pad_mode="constant",):
                #  offset_3d_xy_dx=0.4,
                #  offset_3d_xy_dy=0.4,):

        super(PerspectiveConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.mode = mode
        self.offset_3d = offset_3d
        print(f"self.mode = {self.mode}")
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
        
        ##########################################
        ### Get perspective convolution offset ###
        ##########################################
        # TODO Maybe One day i need to use parameter to save D_RATIO and make it capable to tackle different scale feature map 
        D_RATIO = 16 # Downsamle ratio
        # Import type A calibration file
        # TODO, custumize P2
        self.P2 = kitti_calib_file_parser(os.path.join(CAR_DIR, f"000169.txt"),
                                          new_shape_tf = (288, 1280), 
                                          crop_tf = 100)

        offset = np.zeros((8, 18, 18, 80)) # TODO this hard-wired parameter makes me very uncomfortable
        for v_f_idx in range(18):
            for u_f_idx in range(80):
                v, u = v_f_idx * D_RATIO, u_f_idx * D_RATIO
                slope = get_slope((u, v, AVG_Y3D_CENTER), self.P2)
                
                if self.mode == '2d_offset': # Use Fix 2d offset
                    dcx = 32
                    dcy = 32
                    dsx = dcy - sqrt(1 + slope**2)
                    for i, i_o in enumerate( [-dsx*slope, -dcx-dsx,   -dsx*slope, -dsx,   -dsx*slope, dcx-dsx,
                                               0        , -dcx    ,    0  , 0         ,    0,         dcx    ,
                                               dsx*slope, -dcx+dsx,    dsx*slope,  dsx,    dsx*slope, dcx+dsx ] ):
                        offset[:, i, v_f_idx, u_f_idx] = i_o/D_RATIO

                elif self.mode == '3d_offset_xz':
                    dx = 0.4 # m
                    dz = 1 # m
                    x, y, z = uvy_2_xyz((u, v, AVG_Y3D_CENTER), self.P2)
                    for i, (xi, yi, zi) in enumerate([(x-dx, y, z+dz), (x, y, z+dz), (x+dx, y, z+dz),
                                                      (x-dx, y, z   ), (x, y, z   ), (x+dx, y, z   ),
                                                      (x-dx, y, z-dz), (x, y, z-dz), (x+dx, y, z-dz)]):
                        ui, vi = xyz_2_uv((xi, yi, zi), self.P2)
                        offset[:, i*2  , v_f_idx, u_f_idx] = (vi-v)/D_RATIO
                        offset[:, i*2+1, v_f_idx, u_f_idx] = (ui-u)/D_RATIO
                
                elif self.mode == '3d_offset_yz':
                    dy = 0.5 # m
                    dz = 1.0 # m
                    x, y, z = uvy_2_xyz((u, v, AVG_Y3D_CENTER), self.P2)
                    for i, (xi, yi, zi) in enumerate([(x, y-dy, z-dz), (x, y-dy, z), (x, y-dy, z+dz),
                                                      (x, y   , z-dz), (x, y   , z), (x, y   , z+dz),
                                                      (x, y+dy, z-dz), (x, y+dy, z), (x, y+dy, z+dz)]):
                        ui, vi = xyz_2_uv((xi, yi, zi), self.P2)
                        offset[:, i*2  , v_f_idx, u_f_idx] = (vi-v)/D_RATIO
                        offset[:, i*2+1, v_f_idx, u_f_idx] = (ui-u)/D_RATIO

                elif self.mode == '3d_offset_xy':
                    dx = self.offset_3d
                    dy = self.offset_3d
                    
                    x, y, z = uvy_2_xyz((u, v, AVG_Y3D_CENTER), self.P2)
                    for i, (xi, yi, zi) in enumerate([(x-dx, y-dy, z), (x, y-dy, z), (x+dx, y-dy, z),
                                                      (x-dx, y   , z), (x, y   , z), (x+dx, y   , z),
                                                      (x-dx, y+dy, z), (x, y+dy, z), (x+dx, y+dy, z)]):
                        ui, vi = xyz_2_uv((xi, yi, zi), self.P2)
                        offset[:, i*2  , v_f_idx, u_f_idx] = (vi-v)/D_RATIO
                        offset[:, i*2+1, v_f_idx, u_f_idx] = (ui-u)/D_RATIO
                else:
                    print(f"Not support mode = {self.mode}")
                    raise NotImplementedError
                
                # add offset from regular convolution
                offset[:, :, v_f_idx, u_f_idx] -= np.array([-1,-1,  -1,0,  -1,1,
                                                             0,-1,   0,0,   0,1,
                                                             1,-1,   1,0,   1,1])
        self.offset = torch.from_numpy(offset).type(torch.float32).cuda()

    def forward(self, inputs):
        # B = x.shape[0]
        x  = inputs['features']
        P2 = inputs['P2']
        B = x.shape[0]
        
        # TODO generate different offset according to each P2
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=self.offset[:B, :, :, :],
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=None,
                                          stride=self.stride,)
        return x

