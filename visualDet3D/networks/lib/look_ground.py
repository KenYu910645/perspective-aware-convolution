import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
    One way to improve or interprete the former function is that:
        the network learns a displacement towards the bottom of the feature map -> sample features there
"""

class LookGround(nn.Module):
    """Some Information about LookGround"""
    def __init__(self, input_features, baseline=0.54, relative_elevation=1.65):
        super(LookGround, self).__init__()
        self.disp_create = nn.Sequential(
            nn.Conv2d(input_features, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.extract = nn.Conv2d(1 + input_features, input_features, 1)
        self.extract_no_dis = nn.Conv2d(input_features, input_features, 1) # For NA_NDIS
        
        self.baseline = baseline
        self.relative_elevation = relative_elevation
        self.alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, inputs):

        x = inputs['features']
        P2 = inputs['P2']
        P2 = P2.clone()
        P2[:, 0:2] /= 16.0 # downsample to 16.0

        disp = self.disp_create(x)
        disp = 0.1 * (0.05 * disp + 0.95 * disp.detach()) # It's to cut disp from back proprogation

        batch_size, _, height, width = x.size()

        # Create Disparity Map
        h, w = x.shape[2], x.shape[3]
        x_range = np.arange(h, dtype=np.float32)
        y_range = np.arange(w, dtype=np.float32)
        _, yy_grid  = np.meshgrid(y_range, x_range)

        yy_grid =  x.new(yy_grid).unsqueeze(0) #[1, H, W]
        fy =  P2[:, 1:2, 1:2] #[B, 1, 1]
        cy =  P2[:, 1:2, 2:3] #[B, 1, 1]
        Ty =  P2[:, 1:2, 3:4] #[B, 1, 1]
        
        # Equation (4)
        disparity = fy * self.baseline  * (yy_grid - cy) / (torch.abs(fy * self.relative_elevation + Ty) + 1e-10)
        disparity = F.relu(disparity)
        
        # # Print out disparity image
        # img_dis = disparity[0].cpu().numpy()
        # img_dis = cv2.normalize(img_dis, None, alpha=0,beta=250, norm_type=cv2.NORM_MINMAX)
        # cv2.imwrite("dis_imag.png", img_dis)
        
        # Original coordinates of pixels
        x_base = torch.linspace(-1, 1, width).repeat(batch_size,
                    height, 1).type_as(x)
        y_base = torch.linspace(-1, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(x)

        # Apply shift in Y direction
        h_mean = 1.535
        y_shifts_base = F.relu(
            h_mean * (yy_grid - cy) / (2 * (self.relative_elevation - 0.5 * h_mean))
        ) / (yy_grid.shape[1] * 0.5) # [1, H, W]
        y_shifts = y_shifts_base + disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base, y_base + y_shifts), dim=3)

        ## Output flow field of ground offset for visualization 
        # with open('flow_field.pkl', 'wb') as f:
        #     pickle.dump(flow_field, f)
        #     print(f"Write flow_field to flow_field.pkl")
        #     import time
        #     time.sleep(1)
        # print(f"flow_field = {flow_field.shape}") # [8, 18, 80, 2]
        
        features = torch.cat([disparity.unsqueeze(1), x], dim=1) # [8, 1025, 18, 80]
        output = F.grid_sample(features, flow_field, mode='bilinear',
                        padding_mode='border', align_corners=True)
        return F.relu(x + self.extract(output) * self.alpha)