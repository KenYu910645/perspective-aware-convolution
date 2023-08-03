from typing import Callable, List, Union
import numpy as np
from easydict import EasyDict
from visualDet3D.networks.utils.registry import AUGMENTATION_DICT
from visualDet3D.data.kittidata import KittiObj

@AUGMENTATION_DICT.register_module
class AugmentataionComposer(object):
    """
    Composes a set of functions which take in an image and an object, into a single transform
    """
    def __init__(self, aug_list:List[EasyDict], is_return_all = False):
        self.transforms:List[Callable] = []
        self.is_return_all = is_return_all

        for item in aug_list:
            self.transforms.append(AUGMENTATION_DICT[item.type_name](**item.keywords))

    # @classmethod
    # def from_transforms(cls, transforms:List[Callable]): 
    #     instance:AugmentataionComposer = cls(aug_list=[])
    #     instance.transforms = transforms
    #     return instance

    def __call__(self, left_image:np.ndarray,
                       right_image:Union[None, np.ndarray]=None,
                       p2:Union[None, np.ndarray]=None,
                       p3:Union[None, np.ndarray]=None,
                       labels:Union[None, List[KittiObj]]=None,
                       image_gt:Union[None, np.ndarray]=None,
                       lidar:Union[None, np.ndarray]=None,
                       depth_map:Union[None, np.ndarray]=None,)->List[Union[None, np.ndarray, List[KittiObj]]]:

        for t in self.transforms:
            if hasattr(t, 'use_scene_aware'): # It's CopyPaste object
                left_image, right_image, p2, p3, labels, image_gt, lidar = t(left_image, right_image, p2, p3, labels, image_gt, lidar, depth_map)
            else:
                left_image, right_image, p2, p3, labels, image_gt, lidar = t(left_image, right_image, p2, p3, labels, image_gt, lidar)
            return_list = [left_image, right_image, p2, p3, labels, image_gt, lidar]
        
        if self.is_return_all:
            return return_list
        else:
            return [item for item in return_list if item is not None]
