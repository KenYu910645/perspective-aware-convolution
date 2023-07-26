import torch.nn as nn
import torch.nn.functional as F
import torch

class DepthAwareSample(nn.Module):
    def __init__(self, num_feats_in, num_feats_out, num_output_channel, kernel_size=3, padding=1):
        super().__init__()
        
        self.num_output_channel = num_output_channel
        self.conv_stride_1 = nn.Conv2d(num_feats_in, num_feats_out, stride=1, kernel_size=kernel_size, padding=padding)
        self.conv_stride_2 = nn.Conv2d(num_feats_in, num_feats_out, stride=2, kernel_size=kernel_size, padding=padding)
        self.conv_stride_4 = nn.Conv2d(num_feats_in, num_feats_out, stride=4, kernel_size=kernel_size, padding=padding)

        #
        self.conv_stride_1.weight.data.fill_(0)
        self.conv_stride_1.bias.data.fill_(0)
        self.conv_stride_2.weight.data.fill_(0)
        self.conv_stride_2.bias.data.fill_(0)
        self.conv_stride_4.weight.data.fill_(0)
        self.conv_stride_4.bias.data.fill_(0)


    def forward(self, x):
        b, c, h, w = x.size() # (8, 512, 36, 160)
        
        feature_8  = self.conv_stride_1(x) # (8, 512, 36, 160)
        feature_16 = self.conv_stride_2(x) # (8, 512, 18,  80)
        feature_32 = self.conv_stride_4(x) # (8, 512, 9 ,  40)
        
        # Only crop the ROI rows
        feature_8  = feature_8 [:, :, 6:14, :]
        feature_16 = feature_16[:, :, 7:9, :]
        feature_32 = feature_32[:, :, 4:8 , :]
        #
        # print(f"feature_8  = {feature_8.shape}")  # [8, 64, 10, 160]
        # print(f"feature_16 = {feature_16.shape}") # [8, 64, 5 ,  80]
        # print(f"feature_32 = {feature_32.shape}") # [8, 64, 3 ,  40]
        
        # Anchor Flatten
        feature_8  = feature_8.permute(0, 2, 3, 1)
        feature_8  = feature_8.contiguous().view(b, -1, self.num_output_channel)
        feature_16 = feature_16.permute(0, 2, 3, 1)
        feature_16 = feature_16.contiguous().view(b, -1, self.num_output_channel)
        feature_32 = feature_32.permute(0, 2, 3, 1)
        feature_32 = feature_32.contiguous().view(b, -1, self.num_output_channel)

        #
        # print(f"feature_8  = {feature_8.shape}") # [8, 51200, 2]
        # print(f"feature_16 = {feature_16.shape}") # [8, 12800, 2]
        # print(f"feature_32 = {feature_32.shape}") # [8, 3840, 2]
        
        # Concate the feature
        out_feature = torch.cat((feature_8, feature_16, feature_32), dim=1)
        # print(f"out_feature = {out_feature.shape}") # [8, 67840, 2]
        return out_feature
    
        
class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding

        self.conv = nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1, groups=num_rows)

    def forward(self, x):

        b, c, h, w = x.size() # (8, 512, 36, 160)

        if self.pad: x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

        print(x.shape) # [8, 512, 38, 162]
        
        x_8 = x[:, :, 4:14, :]
        
        # t = int(h / self.num_rows)
        # unfold by rows
        x = x.unfold(dim = 2, size = t + self.pad*2, step = t)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(b, c * self.num_rows, t + self.pad*2, (w+self.pad*2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.conv(x)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w)

        return y