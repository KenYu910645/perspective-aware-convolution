import numpy as np
import torch.nn as nn
import torch
from visualDet3D.networks.utils import DETECTOR_DICT
from visualDet3D.networks.detectors.yolomono3d_core import YoloMono3DCore
from visualDet3D.networks.heads.detection_3d_head import GroundAwareHead
from visualDet3D.networks.heads.retinanet_3d_head import RetinaNet3DHead_BevAnk, RetinaNet3DHead_GACAnk
from visualDet3D.networks.heads.bev_ank_head import BevAnk3DHead
from visualDet3D.networks.detectors.retinanet_2d import RetinaNet3DCore

@DETECTOR_DICT.register_module
class Yolo3D(nn.Module):
    """
        YoloMono3DNetwork
    """
    def __init__(self, network_cfg):
        super(Yolo3D, self).__init__()

        self.obj_types = network_cfg.obj_types
        
        self.exp = network_cfg.exp
        
        self.is_das = getattr(network_cfg, 'is_das', False)
        
        print(f"Yolo3D experiment setting = {self.exp}")
        print(f"self.is_das = {self.is_das}")
        # network_cfg.head.is_two_stage = getattr(network_cfg.head, 'is_two_stage', False)
        
        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

        self.network_cfg.head.is_seperate_cz = getattr(network_cfg.head, 'is_seperate_cz', False)
        print(f"network_cfg.head.is_seperate_cz = {network_cfg.head.is_seperate_cz}")
        
        self.is_writen_anchor_file = False
        
        if self.is_das:
            self.lxl_conv_1024 = nn.Conv2d(in_channels =  512, out_channels = 1024, kernel_size = 1)
            self.fpn_conv      = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, padding=1)
    
    def build_core(self, network_cfg):
        self.core = YoloMono3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = AnchorBasedDetection3DHead(
            **(network_cfg.head)
        )

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
        features  = self.core(dict(image=img_batch, P2=P2)) # [8, 1024, 18, 80]
        
        # if self.is_das: # Neck
        #     # print(features[0].shape) # [8, 512, 36, 160]
        #     # print(features[1].shape) # [8, 1024, 18, 80]
            
        #     # Change number of channel
        #     features_8 = self.lxl_conv_1024(features[0])
            
        #     # Upsmaple 1/16 feature
        #     feature_16 = torch.nn.functional.interpolate(features[1], scale_factor=2, mode='nearest')
            
        #     features = self.fpn_conv(features_8 + feature_16)
        #     # print(f"is_das features = {features.shape}") # [8, 1024, 36, 160]
        
        # For Experiment C
        if self.exp == "C":
            # print(f"features.shape = {features.shape}")
            grid = np.stack(self.build_tensor_grid([features.shape[2], features.shape[3]]), axis=0) #[2, h, w]
            grid = features.new(grid).unsqueeze(0).repeat(features.shape[0], 1, 1, 1) #[1, 2, h, w]
            features = torch.cat([features, grid], dim=1)
            # print(f"features.shape = {features.shape}")

        # cls_preds, reg_preds, cz_preds = self.bbox_head(dict(features=features, P2=P2, image=img_batch))
        preds = self.bbox_head(dict(features=features, P2=P2, image=img_batch))
        anchors = self.bbox_head.get_anchor(img_batch, P2)
        
        # Output Anchor to file
        if (not self.is_writen_anchor_file) and self.network_cfg.head.data_cfg.is_overwrite_anchor_file:
            import pickle
            with open(f"{self.network_cfg.head.data_cfg.anchor_mean_std_path}_anchor.pkl", 'wb') as f:
                pickle.dump(anchors, f)
                print(f"Save to {self.network_cfg.head.data_cfg.anchor_mean_std_path}_anchor.pkl")
            self.is_writen_anchor_file = True
        
        # print(f"[anchor.py] anchor means = {anchors_z.mean()}") # torch.Size([1, 46080])
        # print(f"anchors['anchors'] = {anchors['anchors'].shape}") # [1, 46080, 4]
        # print(f"anchors['mask'] = {anchors['mask'].shape}") # [1, 46080]
        # print(f"anchors['anchor_mean_std_3d'] = {anchors['anchor_mean_std_3d'].shape}") # [46080, 1, 6, 2], z, sin(\t), cos(\t)
        # print(f"valid anchor = {torch.count_nonzero(anchors['mask'])}") # 3860
        # 
        # loss_dict = self.bbox_head.loss(cls_preds, reg_preds, cz_preds, anchors, annotations, P2)
        loss_dict = self.bbox_head.loss(preds, anchors, annotations, P2)
        return loss_dict

    def build_tensor_grid(self, shape):
        """
            For CoordConv exp C
            input:
                shape = (h, w)
            output:
                yy_grid = (h, w)
                xx_grid = (h, w)
        """
        h, w = shape[0], shape[1]
        x_range = np.arange(h, dtype=np.float32)
        y_range = np.arange(w, dtype=np.float32)
        yy, xx  = np.meshgrid(y_range, x_range)
        yy_grid = 2.0 * yy / float(w) - 1 # Make sure value is [-1, 1]
        xx_grid = 2.0 * xx / float(h) - 1
        return yy_grid, xx_grid

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

        features  = self.core(dict(image=img_batch, P2=P2))

        # For experiment C
        if self.exp == "C":
            grid = np.stack(self.build_tensor_grid([features.shape[2], features.shape[3]]), axis=0) #[2, h, w]
            grid = features.new(grid).unsqueeze(0).repeat(features.shape[0], 1, 1, 1) #[1, 2, h, w]
            features = torch.cat([features, grid], dim=1)

        # cls_preds, reg_preds, cz_preds = self.bbox_head(dict(features=features, P2=P2))
        preds   = self.bbox_head(dict(features=features, P2=P2))
        anchors = self.bbox_head.get_anchor(img_batch, P2)
        # print(f"anchors['anchors'] = {anchors['anchors'].shape}") # [1, 46080, 4]
        # print(f"anchors['mask'] = {anchors['mask'].shape}") # [1, 46080]
        # print(f"anchors['anchor_mean_std_3d'] = {anchors['anchor_mean_std_3d'].shape}") # [46080, 1, 6, 2]

        # if self.network_cfg.is_fpn_debug:
        #     cls_preds = cls_preds[:, 184320:230400, :]
        #     reg_preds = reg_preds[:, 184320:230400, :]
        #     anchors['anchors'] = anchors['anchors'][:, 184320:230400, :]
        #     anchors['mask'] = anchors['mask'][:, 184320:230400]
        #     anchors['anchor_mean_std_3d'] = anchors['anchor_mean_std_3d'][184320:230400, :, :, :]
        
        # TODO, need to write some code to transform noam prediction to real thing 
        scores, bboxes, cls_indexes, noam = self.bbox_head.get_bboxes(preds, anchors, P2, img_batch)
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

@DETECTOR_DICT.register_module
class GroundAwareYolo3D(Yolo3D):
    """Some Information about GroundAwareYolo3D"""

    def build_head(self, network_cfg):
        self.bbox_head = GroundAwareHead(
            **(network_cfg.head)
        )

# Add to register BevAnkYolo3D
@DETECTOR_DICT.register_module
class BevAnkYolo3D(Yolo3D):
    """Some Information about BevAnkYolo3D"""

    def build_head(self, network_cfg):
        self.bbox_head = BevAnk3DHead(
            **(network_cfg.head)
        )

# Add to register RetinaNet3DHead_BevAnk
@DETECTOR_DICT.register_module
class RetinaNet3D_BevAnk(Yolo3D):
    def build_head(self, network_cfg): # Use RetinaNet3DHead or GroundAwareHead
        self.bbox_head = RetinaNet3DHead_BevAnk(
            **(network_cfg.head)
        )
    def build_core(self, network_cfg):
        self.core = RetinaNet3DCore(network_cfg.backbone,
                                    network_cfg.neck)

@DETECTOR_DICT.register_module
class RetinaNet3D_GACAnk(Yolo3D):
    def build_head(self, network_cfg):
        self.bbox_head = RetinaNet3DHead_GACAnk(
            **(network_cfg.head)
        )
    def build_core(self, network_cfg):
        self.core = RetinaNet3DCore(network_cfg.backbone,
                                    network_cfg.neck)
