from visualDet3D.networks.lib.pac import PerspectiveConv2d
import torch
import torch.nn as nn

class PAC_module(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        '''
        mode = 2d_offset ,......
        '''
        super(PAC_module, self).__init__()
        self.pac_d16 = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, mode, offset_2d=(16, 16), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), adpative_P2=True),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),) #TODO nn.ReLU(inplace=True)?
        
        self.pac_d32 = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, mode, offset_2d=(32, 32), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), adpative_P2=True),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),)
        
        self.pac_d64 = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, mode, offset_2d=(64, 64), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), adpative_P2=True),
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
        # print(f"cat_features = {cat_features.shape}")
        
        return self.lxl_conv_out(cat_features)
        
        
class PAC_3D_module(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        '''
        mode = 2d_offset ,......
        '''
        super(PAC_3D_module, self).__init__()
        self.pac_dx = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, mode, offset_2d=(64, 32), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), adpative_P2=True),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),)
        
        self.pac_dy = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, mode, offset_2d=(32, 64), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), adpative_P2=True, lock_theta_ortho=True),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),)
        
        self.pac_dz = nn.Sequential(PerspectiveConv2d(in_channels, in_channels//2, mode, offset_2d=(32, 64), kernel_size=3, stride=1, padding=1, bias=False, input_shape=(18,80), adpative_P2=True),
                                     nn.BatchNorm2d(in_channels//2),
                                     nn.ReLU(),)

        self.lxl_conv_out = nn.Sequential(nn.Conv2d(in_channels*5//2, out_channels, kernel_size=1,),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(),)

    def forward(self, inputs):
        pac_dx_feature = self.pac_dx(inputs)
        pac_dy_feature = self.pac_dy(inputs)
        pac_dz_feature = self.pac_dz(inputs)
        
        cat_features = torch.cat([pac_dx_feature, pac_dy_feature, pac_dz_feature, inputs['features']], 1)
        return self.lxl_conv_out(cat_features)
        