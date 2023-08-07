import torch
import torch.nn as nn
import torchvision.ops
import numpy as np
from math import atan2, pi, sin, cos

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

def uvy_2_xyz(uvy, P2):
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
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 input_shape=(18,80),
                 d_rate_xy=(32, 32),
                 lock_theta_ortho=False,):

        super(PerspectiveConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.d_rate_xy = d_rate_xy
        self.h, self.w = input_shape
        self.D_RATIO = 16 # Downsamle ratio
        self.pad_size = 6
        self.lock_theta_ortho = lock_theta_ortho
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

        self.offset_cache = {}
        

    def get_offset(self, P2):
        # Use cache to speed up
        if str(P2) in self.offset_cache: return self.offset_cache[str(P2)]
        
        # offset = np.zeros((8, 18, self.h+self.pad_size*2, self.w+self.pad_size*2))
        offset = np.zeros((1, 18, self.h, self.w))
        
        for v_f_idx in range(self.h):
            for u_f_idx in range(self.w):
                v, u = v_f_idx*self.D_RATIO, u_f_idx*self.D_RATIO
                # 
                slope = get_slope((u, v, AVG_Y3D_CENTER), P2)
                
                # Use Fix 2d offset
                dcx, dcy = self.d_rate_xy
                if self.lock_theta_ortho: theta = pi/2
                else:                     theta = atan2(slope, 1)
                
                dx, dy = (dcy*cos(theta), dcy*sin(theta))
                for i, i_o in enumerate([-dy, -dcx-dx,   -dy, -dx,   -dy, dcx-dx,
                                           0, -dcx   ,     0,   0,     0, dcx   ,
                                          dy, -dcx+dx,    dy,  dx,    dy, dcx+dx]):
                    offset[:, i, v_f_idx, u_f_idx] = i_o/self.D_RATIO
                
                # offset from regular convolution
                offset[:, :, v_f_idx, u_f_idx] -= np.array([-1,-1,  -1,0,  -1,1,
                                                             0,-1,   0,0,   0,1,
                                                             1,-1,   1,0,   1,1])
        offset = torch.from_numpy(offset).type(torch.float32).cuda()
        
        # Update Cache
        self.offset_cache[str(P2)] = torch.clone(offset)
        print(f"Update offset cache")
        # print(P2)
        # print(offset)
        
        return offset
    
    def forward(self, inputs):
        # TODO acceleration?
        x  = inputs['features']
        P2 = inputs['P2']
        B = x.shape[0]
        
        # Get offset
        offsets = torch.cat([ self.get_offset(P2[i].cpu().numpy()) for i in range(B) ], dim=0) # [8, 18, 18, 80]

        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offsets,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=None,
                                          stride=self.stride,)
        return x

class PAC_module(nn.Module):
    def __init__(self, in_channels, out_channels, lock_theta_ortho):
        '''
        mode = 2d_offset ,......
        '''
        super(PAC_module, self).__init__()
        self.pac_d16 = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, d_rate_xy=(16, 16), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), lock_theta_ortho = lock_theta_ortho),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),)
        
        self.pac_d32 = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, d_rate_xy=(32, 32), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), lock_theta_ortho = lock_theta_ortho),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),)
        
        self.pac_d64 = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, d_rate_xy=(64, 64), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), lock_theta_ortho = lock_theta_ortho),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),)

        self.lxl_conv_out = nn.Sequential(nn.Conv2d(in_channels*5//2, out_channels, kernel_size=1,),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(),)

    def forward(self, inputs):
        pac_d16_feature = self.pac_d16(inputs)
        pac_d32_feature = self.pac_d32(inputs)
        pac_d64_feature = self.pac_d64(inputs)
        cat_features = torch.cat([pac_d16_feature, pac_d32_feature, pac_d64_feature, inputs['features']], 1)
        
        return self.lxl_conv_out(cat_features)
