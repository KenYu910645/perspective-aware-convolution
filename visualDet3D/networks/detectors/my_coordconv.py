import numpy as np
import torch.nn as nn
import torch

class MyCoordConv(nn.Module):
    # CoordinateConv: https://arxiv.org/pdf/1807.03247.pdf    

    def __init__(self):
        super(MyCoordConv, self).__init__()
        # TODO, should experiment about out_channels
        self.coordConv = nn.Conv2d(in_channels = 5, out_channels = 3, kernel_size = 3, stride = 1, padding =1)

    def forward(self, x):
        # print(f"x.shape = {x.shape}")

        self.grid = np.stack(self.build_tensor_grid([x.shape[2], x.shape[3]]), axis=0) #[2, h, w]
        
        self.grid = x.new(self.grid).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) #[1, 2, h, w]
        # print(f"self.grid = {self.grid.shape}")

        x = torch.cat([x, self.grid], dim=1)
        # print(f"Before coordconv x = {x.shape}") # [2, 5, 288, 1280]
        x = self.coordConv(x)
        # print(f"After coordconv x = {x.shape}")
        return x

    def build_tensor_grid(self, shape):
        """
            input:
                shape = (h, w)
            output:
                yy_grid = (h, w)
                xx_grid = (h, w)
        """
        h, w = shape[0], shape[1]
        # print((h, w)) # (288, 1280)
        x_range = np.arange(h, dtype=np.float32)
        y_range = np.arange(w, dtype=np.float32)
        yy, xx  = np.meshgrid(y_range, x_range)
        # yy_grid = 2.0 * yy / float(h) - 1 # I think this is wrong implementation
        # xx_grid = 2.0 * xx / float(w) - 1
        yy_grid = 2.0 * yy / float(w) - 1 # Make sure value is [-1, 1]
        xx_grid = 2.0 * xx / float(h) - 1
        return yy_grid, xx_grid


if __name__ == "__main__":
    from torchsummary import summary
    network = MyCoordConv().cuda()
    summary(network, (3, 288, 1280)) # 8, 3, 288, 1280
