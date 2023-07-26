import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import cv2 

"""
    One way to improve or interprete the former function is that:
        the network learns a displacement towards the bottom of the feature map -> sample features there
"""

class LookGround(nn.Module):
    """Some Information about LookGround"""
    def __init__(self, input_features, exp, baseline=0.54, relative_elevation=1.65):
        super(LookGround, self).__init__()
        self.exp = exp
        print(f"LookGround's __init__() self.exp = {self.exp}")
        self.disp_create = nn.Sequential(
            nn.Conv2d(input_features, 1, 3, padding=1),
            nn.Tanh(),
        )
        # For WGAC
        self.disp_up_create = nn.Sequential(
            nn.Conv2d(input_features, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.disp_down_create = nn.Sequential(
            nn.Conv2d(input_features, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.disp_left_create = nn.Sequential(
            nn.Conv2d(input_features, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.disp_right_create = nn.Sequential(
            nn.Conv2d(input_features, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.extract = nn.Conv2d(1 + input_features, input_features, 1)
        self.extract_no_dis = nn.Conv2d(input_features, input_features, 1) # For NA_NDIS
        
        self.baseline = baseline
        self.relative_elevation = relative_elevation
        self.alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        # For NA_WGAC
        self.alpha_left  = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.alpha_right = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.alpha_up    = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.alpha_down  = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        # For NA_WGAC, slope tensor
        self.slope = None # 
        # For NA_WGAC, concatenation
        self.lxl_Conv = nn.Conv2d(6149, 1024, 1, bias=True) # 1x1 convolution
        self.BASE_OFFSET = 0

    def forward(self, inputs):
        x = inputs['features']
        P2 = inputs['P2']
        # Init self.slope
        if self.exp == "NA_WGAC" and self.slope == None:
            self.slope = torch.zeros(18, 80, device="cuda")
            for v in range(18):
                for u in range(80):
                    self.slope[v, u] = self.get_slope((u, v), P2[0].cpu().numpy(), (18, 80))
            # Init self.slope_base_offset
            self.slope_base_offset = torch.sqrt((self.BASE_OFFSET**2)/torch.square(1+self.slope))
            self.slope_base_offset = torch.clip(self.slope_base_offset, min=-self.BASE_OFFSET, max=self.BASE_OFFSET)
        
        P2 = P2.clone()
        P2[:, 0:2] /= 16.0 # downsample to 16.0

        disp = self.disp_create(x)
        disp = 0.1 * (0.05 * disp + 0.95 * disp.detach()) # It's to cut disp from back proprogation
        # print(f"disp.shape = {disp.shape}") # [8, 1, 18, 80]
        if self.exp == "NA_WGAC": #TODO use detach to cut the propogation? 
            disp_up = self.disp_up_create(x)
            disp_down = self.disp_down_create(x)
            disp_left = self.disp_left_create(x)
            disp_right = self.disp_right_create(x)

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
        
        if self.exp == "NA_WGAC":
            
            # Apply shift in left direction #TODO, does direction even have any meaning?
            left_shifts = self.BASE_OFFSET + disp_left[:, 0, :, :]
            flow_field_left = torch.stack((x_base - left_shifts, y_base), dim=3)
            # 
            right_shifts = self.BASE_OFFSET + disp_right[:, 0, :, :]
            flow_field_right = torch.stack((x_base + right_shifts, y_base), dim=3)
            #
            up_shifts = self.slope_base_offset + disp_up[:, 0, :, :]
            flow_field_up   = torch.stack((x_base + up_shifts,   y_base + torch.mul(up_shifts, self.slope)), dim=3)
            # 
            down_shifts = -self.slope_base_offset + disp_down[:, 0, :, :]
            flow_field_down = torch.stack((x_base + down_shifts, y_base + torch.mul(down_shifts, self.slope)), dim=3)

        # print(f"flow_field_up = {up_shifts.min()}")
        # print(f"flow_field_up = {up_shifts.max()}")

        # with open('flow_field.pkl', 'wb') as f:
        #     pickle.dump(flow_field, f)
        #     print(f"Write flow_field to flow_field.pkl")
        #     import time
        #     time.sleep(1)
        
        # print(f"flow_field = {flow_field.shape}") # [8, 18, 80, 2]
        # In grid_sample coordinates are assumed to be between -1 and 1
        if self.exp == "no_disparity_map": # Use ground feature but not disparity map
            output = F.grid_sample(x, flow_field, mode='bilinear',
                                padding_mode='border', align_corners=True)
            return F.relu(x + self.extract_no_dis(output) * self.alpha)
        
        elif self.exp == "no_ground_feature": # Use disparity but no ground feature
            output = torch.cat([disparity.unsqueeze(1), x], dim=1) # [8, 1025, 18, 80]
            return F.relu(self.extract(output))
        
        elif self.exp == "NA_WGAC":
            features = torch.cat([disparity.unsqueeze(1), x], dim=1) # [8, 1025, 18, 80]
            f_left  = F.grid_sample(features, flow_field_left,  mode='bilinear', padding_mode='border', align_corners=True)
            f_right = F.grid_sample(features, flow_field_right, mode='bilinear', padding_mode='border', align_corners=True)
            f_up    = F.grid_sample(features, flow_field_up,    mode='bilinear', padding_mode='border', align_corners=True)
            f_down  = F.grid_sample(features, flow_field_down,  mode='bilinear', padding_mode='border', align_corners=True)
            # Element-wise addition, # TODO this is wierd, I should use different extractor, 
            # TODO, add original GAC feature
            # return F.relu(x +\
            #               self.extract(f_left)  * self.alpha_left  +\
            #               self.extract(f_right) * self.alpha_right +\
            #               self.extract(f_up)    * self.alpha_up    +\
            #               self.extract(f_down)  * self.alpha_down  )
            # Concatenation 
            # print(x.shape) # [8, 1024, 18, 80]
            output = torch.cat([x, features, f_left, f_right, f_up, f_down], dim=1)
            # print(output.shape) # [8, 6149, 18, 80]
            return F.relu(self.lxl_Conv(output))

        else:
            features = torch.cat([disparity.unsqueeze(1), x], dim=1) # [8, 1025, 18, 80]
            output = F.grid_sample(features, flow_field, mode='bilinear',
                            padding_mode='border', align_corners=True)
            return F.relu(x + self.extract(output) * self.alpha)
        # print(f"x = {x.shape}") # [8, 1024, 18, 80]
        # print(f"output = {output.shape}") # [8, 1025, 18, 80]
    
    def get_slope(self, uv, P2, reso = (375, 1242)):
        MAX_SLOPE = 500

        # Remap uv to (375, 1242)
        uv = (uv[0]*1242/reso[1], uv[1]*375/reso[0])
        
        x, y, z = self.uv_2_xyz(uv, P2)
        u1, v1 = self.xyz_2_uv((x, y, z-10), P2)
        u2, v2 = self.xyz_2_uv((x, y, z+10), P2)
        
        # Avoid ZeroDivision
        if (u2-u1) == 0 : return MAX_SLOPE
        
        slope = (v2 - v1) / (u2 - u1)
        
        if slope > MAX_SLOPE: 
            return MAX_SLOPE
        elif slope < -MAX_SLOPE:
            return -MAX_SLOPE
        else:
            return slope


    def uv_2_xyz(self, X, P2, y0=1.65):
        P2_3x3 = np.array([P2[0, :3], P2[1, :3], P2[2, :3]])
        P2_inv = np.linalg.inv(P2_3x3)

        alpha = y0/( P2_inv[1,0]*X[0] + P2_inv[1,1]*X[1] + P2_inv[1,2] )
        
        ans = np.matmul(P2_inv, np.array([ [X[0]*alpha], [X[1]*alpha], [alpha]]))
        return (ans[0][0], ans[1][0], ans[2][0])

    def xyz_2_uv(self, X, P2):
        '''
        Transform camera cooridiante(x,y,z) to image plane(u,v)
        '''
        x,y,z = X
        X_img = np.matmul(P2, np.array([[x], [y], [z], [1]]))
        X_img /= X_img[2]
        u, v = int(X_img[0]), int(X_img[1])
        return (u,v)