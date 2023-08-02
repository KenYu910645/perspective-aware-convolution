import torch
import torch.nn as nn
from visualDet3D.networks.utils import DETECTOR_DICT
from visualDet3D.networks.backbones import resnet
from visualDet3D.networks.detectors.yolo3d_head import Yolo3D_Head

@DETECTOR_DICT.register_module
class Yolo3D(nn.Module):
    def __init__(self, cfg):
        super(Yolo3D, self).__init__()

        self.cfg = cfg
        
        # Build backbone
        if "resnext_mode_name" in cfg.detector.backbone:
            print(f"Using ResNext as Backbone")
            resnext_mode = torch.hub.load('pytorch/vision:v0.10.0', 
                                           cfg.detector.backbone["resnext_mode_name"], 
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
            self.backbone = resnet(**cfg.detector.backbone)
        
        
        # Build detection head 
        self.detection_head = Yolo3D_Head(cfg)
        
        # self.is_writen_anchor_file = False
        
        if cfg.detector.head.is_das:
            self.lxl_conv_1024 = nn.Conv2d(in_channels =  512, out_channels = 1024, kernel_size = 1)
            self.fpn_conv      = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, padding=1)

    def backbone_forward(self, x):
        # print(f"x['image'] = {x['image'].shape}") # torch.Size([8, 3, 288, 1280])
        x = self.backbone(x['image']) # [8, 2048, 9, 40]
        if "resnext_mode_name" in self.cfg.detector.backbone:
            x = torch.unsqueeze(x, dim=0)

        if len(x) == 1: x = x[0] # x = torch.Size([8, 1024, 18, 80]

        return x

    def training_forward(self, img_batch, annotations, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """

        # print(f"P2 = {P2}") [8, 3, 4] 
        # print(f"img_batch.shape = {img_batch.shape}") # [8, 3, 288, 1280]

        # Feature Extraction
        features  = self.backbone_forward(dict(image=img_batch, P2=P2)) # [8, 1024, 18, 80]

        # Detection Head
        preds   = self.detection_head(dict(features=features, P2=P2, image=img_batch))
        anchors = self.detection_head.get_anchor(img_batch, P2)
        
        # # Output Anchor to file
        # if (not self.is_writen_anchor_file) and self.detector_cfg.head.data_cfg.is_overwrite_anchor_file:
        #     import pickle
        #     with open(f"{self.detector_cfg.head.data_cfg.anchor_mean_std_path}_anchor.pkl", 'wb') as f:
        #         pickle.dump(anchors, f)
        #         print(f"Save to {self.detector_cfg.head.data_cfg.anchor_mean_std_path}_anchor.pkl")
        #     self.is_writen_anchor_file = True
        
        # print(f"[anchor.py] anchor means = {anchors_z.mean()}") # torch.Size([1, 46080])
        # print(f"anchors['anchors'] = {anchors['anchors'].shape}") # [1, 46080, 4]
        # print(f"anchors['mask'] = {anchors['mask'].shape}") # [1, 46080]
        # print(f"anchors['anchor_mean_std_3d'] = {anchors['anchor_mean_std_3d'].shape}") # [46080, 1, 6, 2], z, sin(\t), cos(\t)
        # print(f"valid anchor = {torch.count_nonzero(anchors['mask'])}") # 3860
        # 
        
        # Loss function
        loss_dict = self.detection_head.loss(preds, anchors, annotations, P2)
        
        return loss_dict

    def test_forward(self, img_batch, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            results: a nested list:
                result[i] = detection_results for obj_types[i]
                    each detection result is a list [scores, bbox, obj_type]:
                        bbox = [bbox2d(length=4) , cx, cy, z, w, h, l, alpha]
        """
        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing
        
        # This is for visulization 
        # import pickle
        # with open('img_batch.pkl', 'wb') as f:
        #     pickle.dump(img_batch, f)
        #     print(f"Write img_batch to img_batch.pkl")

        # Feature Extraction
        features  = self.backbone_forward(dict(image=img_batch, P2=P2)) # [8, 1024, 18, 80]

        # Detection Head
        preds   = self.detection_head(dict(features=features, P2=P2))
        anchors = self.detection_head.get_anchor(img_batch, P2)
        # print(f"anchors['anchors'] = {anchors['anchors'].shape}") # [1, 46080, 4]
        # print(f"anchors['mask'] = {anchors['mask'].shape}") # [1, 46080]
        # print(f"anchors['anchor_mean_std_3d'] = {anchors['anchor_mean_std_3d'].shape}") # [46080, 1, 6, 2]
        
        scores, bboxes, cls_indexes, noam = self.detection_head.get_bboxes(preds, anchors, P2, img_batch)
        # print(f"scores = {scores.shape}") # torch.Size([5]) # [number of detection], confident score
        # print(f"bboxes = {bboxes.shape}") # torch.Size([5, 11]),
        # print(f"cls_indexes = {cls_indexes.shape}") # torch.Size([5]), class_idx
        
        return scores, bboxes, cls_indexes, noam

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, calib = inputs
            return self.training_forward(img_batch, annotations, calib)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch, calib)
