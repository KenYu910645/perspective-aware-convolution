import numpy as np
import torch.nn as nn
import torch
from visualDet3D.networks.backbones import resnet

class YoloMono3DCore(nn.Module):
    def __init__(self, backbone_arguments=dict()):
        super(YoloMono3DCore, self).__init__()
        self.backbone_arguments = backbone_arguments
        
        if "resnext_mode_name" in backbone_arguments:
            print(f"Using ResNext as Backbone")
            resnext_mode = torch.hub.load('pytorch/vision:v0.10.0', 
                                           backbone_arguments["resnext_mode_name"], 
                                           pretrained=True)
            
            # Get a list of child modules
            child_modules = list(resnext_mode.children())

            # Remove the last module in the list
            child_modules.pop() # Remove Linear(in_features=2048, out_features=1000, bias=True)
            child_modules.pop() # Remove AdaptiveAvgPool2d(output_size=(1, 1))
            child_modules.pop() # Remove the final stage of ResNext

            # Reconstruct the model with the remaining modules
            self.backbone = nn.Sequential(*child_modules)
        else:
            self.backbone = resnet(**backbone_arguments)
    
    def forward(self, x):
        # print(f"x['image'] = {x['image'].shape}") # torch.Size([8, 3, 288, 1280])
        x = self.backbone(x['image']) # [8, 2048, 9, 40]
        if "resnext_mode_name" in self.backbone_arguments:
            x = torch.unsqueeze(x, dim=0)

        if len(x) == 1: x = x[0]
        # print(f"x.shape = {x.shape}") # torch.Size([8, 1024, 18, 80]
        return x