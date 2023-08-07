from __future__ import print_function, division
import numpy as np
from copy import deepcopy
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from visualDet3D.utils.cal import theta2alpha_3d, BBox3dProjector
from visualDet3D.utils.kitti_data_parser import KittiObj
from visualDet3D.data_augmentation.augmentation_composer import AugmentataionComposer

class KittiDataset(Dataset):
    def __init__(self, cfg, imdb_frames, split='training'):
        super(KittiDataset, self).__init__()
        
        self.cfg = cfg 

        self.is_copy_paste = any(d['type_name'] == 'CopyPaste' for d in cfg.data.train_augmentation)

        self.is_train = (split == 'training')
        
        self.use_right_image = cfg.data.use_right_image
        print(f"[mono_dataset.py] use_right_image = {self.use_right_image}")
        self.is_reproject = getattr(cfg.data, 'is_reproject', True) # if reproject 2d

        # Load imdb file
        # print(f"cfg.path.preprocessed_path = {cfg.path.preprocessed_path}")
        # imdb_path = os.path.join(cfg.path.preprocessed_path, split, 'imdb.pkl')
        # print(cfg.path.preprocessed_path = )
        # self.imdb = pickle.load( open(imdb_path, 'rb'))
        self.imdb = imdb_frames
        
        self.output_dict = {
                "calib": False,
                "image": True,
                "label": False,
                "velodyne": False,
                "depth": self.is_copy_paste,
            }

        if self.is_train:
            self.transform = AugmentataionComposer(cfg.data.train_augmentation, is_return_all = False)# build_augmentator(cfg.data.train_augmentation)
        else:
            self.transform = AugmentataionComposer(cfg.data.test_augmentation, is_return_all = False) # build_augmentator(cfg.data.test_augmentation)
        
        self.projector = BBox3dProjector()

    def _reproject(self, P2:np.ndarray, transformed_label:List[KittiObj]) -> Tuple[List[KittiObj], np.ndarray]:

        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        for obj in transformed_label:
            obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
        bbox3d_origin = torch.tensor([[obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha] for obj in transformed_label], dtype=torch.float32)
        abs_corner, homo_corner, _ = self.projector(bbox3d_origin, bbox3d_origin.new(P2))
        for i, obj in enumerate(transformed_label):
            extended_center = np.array([obj.x, obj.y - 0.5 * obj.h, obj.z, 1])[:, np.newaxis] #[4, 1]
            image_center = (P2 @ extended_center)[:, 0] #[3]
            image_center[0:2] /= image_center[2]
            bbox3d_state[i] = np.concatenate([image_center,[obj.w, obj.h, obj.l, obj.alpha]]) #[7]

        max_xy, _= homo_corner[:, :, 0:2].max(dim = 1)  # [N,2]
        min_xy, _= homo_corner[:, :, 0:2].min(dim = 1)  # [N,2]

        result = torch.cat([min_xy, max_xy], dim=-1) #[:, 4]

        bbox2d = result.cpu().numpy()

        if self.is_reproject: # Use 3d bbox and 2d bbox relation to reproject
            for i in range(len(transformed_label)):
                transformed_label[i].bbox_l = bbox2d[i, 0]
                transformed_label[i].bbox_t = bbox2d[i, 1]
                transformed_label[i].bbox_r = bbox2d[i, 2]
                transformed_label[i].bbox_b = bbox2d[i, 3]
        
        return transformed_label, bbox3d_state


    def __getitem__(self, index):

        kitti_data = self.imdb[index % len(self.imdb)]
        
        # The calib and label has been preloaded to minimize the time in each indexing
        if index >= len(self.imdb):
            kitti_data.output_dict = {
                "calib": True,
                "image": False,
                "image_3":True,
                "label": False,
                "velodyne": False,
                "depth": self.is_copy_paste,
            }
            calib, _, image, _, _, depth = kitti_data.read_data()
            calib.P2 = calib.P3 # a workaround to use P3 for right camera images. 3D bboxes are the same(cx, cy, z, w, h, l, alpha)
        else:
            kitti_data.output_dict = self.output_dict
            _, image, _, _, depth = kitti_data.read_data()
            calib = kitti_data.calib
        calib.image_shape = image.shape
        label = kitti_data.label # label: list of kittiObj
        label = []
        for obj in kitti_data.label:
            if obj.type in self.cfg.obj_types:
                label.append(obj)

        # label transformation happen here
        transformed_image, transformed_P2, transformed_label = self.transform(image, p2=deepcopy(calib.P2), labels=deepcopy(label), depth_map=depth)
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label) > 0:
            transformed_label, bbox3d_state = self._reproject(transformed_P2, transformed_label)
        # print(f"transformed_label after _reproject = {(transformed_label[0].bbox_l, transformed_label[0].bbox_t, transformed_label[0].bbox_r, transformed_label[0].bbox_b)}")

        bbox2d = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in transformed_label])

        loc_3d_ry = np.array( [[obj.x, obj.y, obj.z, obj.ry] for obj in transformed_label] )
        
        output_dict = {'calib': transformed_P2,
                       'image': transformed_image,
                       'label': [obj.type for obj in transformed_label], 
                       'bbox2d': bbox2d, #[N, 4] [x1, y1, x2, y2]
                       'bbox3d': bbox3d_state, 
                       'original_shape':image.shape,
                       'original_P':calib.P2.copy(),
                       'loc_3d_ry':loc_3d_ry,}
        return output_dict

    def __len__(self):
        if self.is_train and self.use_right_image:
            return len(self.imdb) * 2
        else:
            return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([item["image"] for item in batch])#[batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        calib     = [item["calib"]     for item in batch]
        label     = [item['label']     for item in batch]
        bbox2ds   = [item['bbox2d']    for item in batch]
        bbox3ds   = [item['bbox3d']    for item in batch]
        loc_3d_ry = [item['loc_3d_ry'] for item in batch]
        # This line will cause warning:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        # add np.array() to avoid warning.
        return torch.from_numpy(rgb_images).float(), torch.tensor(np.array(calib)).float(), label, bbox2ds, bbox3ds, loc_3d_ry

class KittiTestDataset(KittiDataset):
    def __init__(self, cfg, imdb_frames, split="test"):
        super(KittiTestDataset, self).__init__(cfg, imdb_frames, split="test")
        self.imdb = imdb_frames
        self.output_dict = {
                "calib": False,
                "image": True,
                "label": False,
                "velodyne": False,
                "depth": False,
        }

    def __getitem__(self, index):

        kitti_data = self.imdb[index % len(self.imdb)]
        kitti_data.output_dict = self.output_dict
        _, image, _, _, _ = kitti_data.read_data()

        calib = kitti_data.calib
        calib.image_shape = image.shape
        transformed_image, transformed_P2 = self.transform(image, p2=deepcopy(calib.P2))
        output_dict = {'calib': transformed_P2,
                       'image': transformed_image,
                       'original_shape':image.shape,
                       'original_P':calib.P2.copy()}
        return output_dict

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([item["image"]
                               for item in batch])  # [batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        calib = [item["calib"] for item in batch]
        return torch.from_numpy(rgb_images).float(), calib 
