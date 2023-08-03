"""
This file contains all PyTorch data augmentation functions.

Every transform should have a __call__ function which takes in (self, image, imobj)
where imobj is an arbitary dict containing relevant information to the image.

In many cases the imobj can be None, which enables the same augmentations to be used
during testing as they are in training.

Optionally, most transforms should have an __init__ function as well, if needed.
"""
import numpy as np
from numpy import random
import cv2
import random as random_pkg
import pickle
import copy
from math import sqrt, pi
from easydict import EasyDict as edict

from visualDet3D.networks.utils.utils import BBox3dProjector
from visualDet3D.utils.utils import theta2alpha_3d
from visualDet3D.networks.utils.registry import AUGMENTATION_DICT
from visualDet3D.data.kittidata import KittiObj
from .augmentation_composer import AugmentataionComposer

# from visualDet3D.data_augmentation.copy_paste import CopyPaste_Object
from visualDet3D.utils.iou_3d import get_3d_box, box3d_iou, box2d_iou, box2d_iog

# Added by spiderkiller
@AUGMENTATION_DICT.register_module
class CopyPaste(object):
    """
    Randomly Copy instance to this image
    """
    def __init__(self, num_add_obj ,use_seg, use_z_jitter ,solid_ratio, use_scene_aware):
        self.num_add_obj     = num_add_obj # 3
        self.use_seg         = use_seg
        self.use_z_jitter    = use_z_jitter
        self.solid_ratio     = solid_ratio
        self.use_scene_aware = use_scene_aware
        
        self.image_dir = "/home/lab530/KenYu/kitti/training/image_2/"
        self.depth_dir = "/home/lab530/KenYu/kitti/training/image_depth/"
        self.instance_pool_path = "/home/lab530/KenYu/visualDet3D/exp_output/scene_aware/Mono3D/output/training/instance_pool.pkl"
        self.imgs_src_path = "/home/lab530/KenYu/visualDet3D/exp_output/scene_aware/Mono3D/output/training/imgs_src.pkl"
        
        # Load Instance Pool
        print(f"Loading instance pool from {self.instance_pool_path}")
        self.instance_pool = pickle.load(open(self.instance_pool_path, "rb"))
        
        # Load src images
        print(f"Loading instance pool from {self.imgs_src_path}")
        self.imgs_src      = pickle.load(open(self.imgs_src_path, "rb"))
        
        # print(f"Loading source images from {self.image_dir}")
        # self.imgs_src  = {fn.split(".")[0]: cv2.imread(os.path.join(self.image_dir, fn)) for fn in os.listdir(self.image_dir)}

    def check_depth_map(self, gt_add, tar_depth):
        try:
            # Calculate the number of pixels above the threshold
            num_above_threshold = np.count_nonzero(tar_depth[gt_add.ymin:gt_add.ymax, gt_add.xmin:gt_add.xmax] < gt_add.cz)
            percent_above_threshold = num_above_threshold / gt_add.area
            
            if percent_above_threshold > 0.25: 
                return False
            else: 
                return True
        except TypeError:
            pass
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    def check_suitable(self, gt_add, labels_tar):
        '''
        Check whether gt_add is sutiable to add to gts_tar
        '''
        for gt_tar in labels_tar:
            # Get 2D IOU, Avoid far object paste on nearer object
            iou_2d = box2d_iou((gt_add.xmin  , gt_add.ymin  , gt_add.xmax  , gt_add.ymax),
                               (gt_tar.bbox_l, gt_tar.bbox_t, gt_tar.bbox_r, gt_tar.bbox_b))
            if iou_2d > 0.0 and gt_add.z3d > gt_tar.z: return False
            
            # IoG check
            b1_iog, b2_iog = box2d_iog((gt_add.xmin, gt_add.ymin, gt_add.xmax, gt_add.ymax),
                                       (gt_tar.bbox_l, gt_tar.bbox_t, gt_tar.bbox_r, gt_tar.bbox_b))
            if max(b1_iog, b2_iog) > 0.7 : return False
            
            # Get 3D IOU, Avoid 3D bbox collide with each other on BEV
            gt_tar_corners_3d = get_3d_box((gt_tar.l, gt_tar.w, gt_tar.h), 
                                            gt_tar.ry, 
                                           (gt_tar.x, gt_tar.y, gt_tar.z))
            try: # TODO this is because I can't solve ZeroDivision in iou_3d.py
                iou_3db, iou_bev = box3d_iou(gt_tar_corners_3d, gt_add.corners_3d)
            except Exception as e:
                print("Error:", str(e))
                iou_3db, iou_bev = (0, 0)
                return False
            if iou_bev > 0.0: return False
        return True

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None, depth_map=None):
        
        assert right_image == None, "Currently don't support right_image augment with copyPaste"

        # Add objects in the target image
        for obj_idx in range(self.num_add_obj):
            random.shuffle(self.instance_pool)
            gt_rst_new = None
            gt_rst_old = None
            for i_gt_add, gt_add in enumerate(self.instance_pool):
                # Don't use object without segmented label
                if self.use_seg and len(gt_add.seg_points) == 0: continue
                #
                gt_add_old = copy.deepcopy( gt_add )
                gt_add_new = copy.deepcopy( gt_add )
                
                # Jitter Z direction
                if self.use_z_jitter:
                    cz_new = random.uniform( gt_add.cz*0.8, gt_add.cz*1.2 )
                    s = gt_add.cz / cz_new
                    
                    d_bev_old = sqrt(gt_add.x3d**2 + gt_add.z3d**2)
                    d_bev_new = sqrt(cz_new**2 - (gt_add.y3d - gt_add.h/2)**2)
                    
                    gt_add_new.x3d *= d_bev_new / d_bev_old
                    gt_add_new.z3d *= d_bev_new / d_bev_old
                    gt_add_new.P2 = p2
                    gt_add_new.reprojection()
                
                # Check whether it's a good spawn by scanning through all the existed groundTrue
                if not self.check_suitable(gt_add_new, labels): continue
                
                # Get aug image
                img_src = self.imgs_src[gt_add_new.idx_img]
                
                img_tar_h, img_tar_w, _ = left_image.shape
                img_src_h, img_src_w, _ = img_src.shape
                
                # Source image
                patch_src = img_src[gt_add_old.ymin:gt_add_old.ymax,
                                    gt_add_old.xmin:gt_add_old.xmax]
                
                patch_tar_w, patch_tar_h = (int(gt_add_new.xmax - gt_add_new.xmin),
                                            int(gt_add_new.ymax - gt_add_new.ymin))

                # Resize source patch to fit the target patch
                patch_src = cv2.resize(patch_src, (patch_tar_w, patch_tar_h), interpolation=cv2.INTER_AREA)
                if self.use_seg:
                    old_h = gt_add_old.ymax - gt_add_old.ymin
                    old_w = gt_add_old.xmax - gt_add_old.xmin
                    for i, (xp, yp) in enumerate(gt_add_new.seg_points):
                        gt_add_new.seg_points[i][0] *= (patch_tar_w/old_w)
                        gt_add_new.seg_points[i][1] *= (patch_tar_h/old_h)

                # 2D bbox saturation and crop image on the bournding
                if gt_add_new.xmin < 0 :
                    patch_src = patch_src[:, abs(gt_add_new.xmin):]
                    gt_add_new.xmin = 0
                if gt_add_new.ymin < 0 :
                    patch_src = patch_src[abs(gt_add_new.ymin):, :]
                    gt_add_new.ymin = 0
                if gt_add_new.xmax > img_tar_w:
                    patch_src = patch_src[:, :img_tar_w-gt_add_new.xmax]
                    gt_add_new.xmax = img_tar_w
                if gt_add_new.ymax > img_tar_h:
                    patch_src = patch_src[:img_tar_h-gt_add_new.ymax, :]
                    gt_add_new.ymax = img_tar_h
                
                # Check depth map
                if self.use_scene_aware and self.check_depth_map(gt_add_new, depth_map): continue

                gt_rst_old = gt_add_old
                gt_rst_new = gt_add_new
                break

            # If couldn't find any augmented image, copy paste image without modification
            if gt_rst_new == None:
                # print(f"[WARNING] Cannot find suitable gt to add")
                break
            
            # Paste the source instance on target image
            if self.use_seg:
                mask = np.zeros_like(patch_src[:, :, 1])
                
                obj_countor = []
                for xp, yp in gt_rst_new.seg_points:
                    obj_countor.append(np.array([[int(xp), int(yp)]], dtype=np.int32))

                cv2.drawContours(mask, np.array([obj_countor]), -1, 255, -1)
                left_image[gt_rst_new.ymin:gt_rst_new.ymax, gt_rst_new.xmin:gt_rst_new.xmax][mask == 255] = patch_src[mask == 255]
            else:
                # Add new object on augmented image and .txt
                left_image[gt_rst_new.ymin:gt_rst_new.ymax, gt_rst_new.xmin:gt_rst_new.xmax] = self.solid_ratio*patch_src + \
                left_image[gt_rst_new.ymin:gt_rst_new.ymax, gt_rst_new.xmin:gt_rst_new.xmax]*(1-self.solid_ratio)
            
            # Add new object's label to source gts
            # label.append(gt_rst_new)
            labels.append(KittiObj(gt_rst_new.__str__()))
        
        return left_image, right_image, p2, p3, labels, image_gt, lidar

# Added by spiderkiller
@AUGMENTATION_DICT.register_module
class EraseBackGround(object):
    """
    Erase all image background. only pixel in 2D bounding box will remain
    """
    def __init__(self):
        pass
        
    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        # print(f"left_image = {left_image.shape}") # (288, 1280, 3)
        # print(f"p2 = {p2.shape}") # (3, 4)
        # print(f"labels = {len(labels)}") # labels = 0
        if right_image != None:
            raise NotImplementedError("use right_image with EraseBackGround augmentation current not implemented")
        
        left_image_new = np.zeros(left_image.shape)
        if labels and isinstance(labels, list):
            for obj in labels:
                x1, y1, x2, y2 = (int(obj.bbox_l), int(obj.bbox_t), int(obj.bbox_r), int(obj.bbox_b))
                left_image_new[y1:y2, x1:x2 , :] = left_image[y1:y2, x1:x2 , :]
        left_image = left_image_new
                
        return left_image, right_image, p2, p3, labels, image_gt, lidar

# Added by spiderkiller
@AUGMENTATION_DICT.register_module
class RandomZoom(object):
    """
    Randomly zoom in
    """
    def __init__(self, scale_range):
        self.s_low_bound, self.s_up_bound = scale_range # Scaling factor

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        
        h, w, c = left_image.shape # (288, 1280, 3)
        
        cu = p2[0, 2]
        cv = p2[1, 2]
        
        # Randomly generate scale in range
        s = random_pkg.uniform(self.s_low_bound, self.s_up_bound)
        
        left_image_new_scale = cv2.resize(left_image, (0,0), fx=s, fy=s)
        if s > 1.0: # Crop In 
            left_image = left_image_new_scale[int(cv*s - cv) : int(cv*s - cv) + h,
                                              int(cu*s - cu) : int(cu*s - cu) + w]
        else:# Pad out
            crop_x1, crop_y1, crop_x2, crop_y2 = ( int(cu*s - cu), 
                                                   int(cv*s - cv), 
                                                   int(cu*s - cu) + w, 
                                                   int(cv*s - cv) + h )
            # If crop region extends beyond image boundaries, adjust crop region and zero-pad image
            left_image = np.pad(left_image_new_scale, ((abs(crop_y1), crop_y2-left_image_new_scale.shape[0]), 
                                                       (abs(crop_x1), crop_x2-left_image_new_scale.shape[1]), 
                                                        (0,0)), mode='constant')

        # Deal with right_image
        if right_image is not None:
            raise NotImplementedError
            right_image = right_image[int(h/2 - h/(2*self.s)) : int(h/2 + h/(2*self.s)),
                                      int(w/2 - w/(2*self.s)) : int(w/2 + w/(2*self.s))]
            right_image = cv2.resize(right_image, (w, h), interpolation=cv2.INTER_AREA)
        
        # Deal with labels
        # 'alpha', 'bbox_b', 'bbox_l', 'bbox_r', 'bbox_t', 'h', 'l', 'occluded', 'ry', 'score', 'truncated', 'type', 'w', 'x', 'y', 'z'
        if labels:
            if isinstance(labels, list):
                # 2d bbox label will be take care at in visualDet3D/data/kitti/dataset/mono_dataset.py
                # it reproject 3d bbox back it image plane to get 2d bbox.
                # it's no need to jusify alpha as well, because mono_dataset.py do that when reproject option is on
                
                # TODO, i think it should be cz/s, not z3d BUG!!!!!
                for obj in labels:
                    dz = obj.z*(1-1/s)
                    obj.z -= dz
        
        return left_image, right_image, p2, p3, labels, image_gt, lidar

# Added by spiderkiler
@AUGMENTATION_DICT.register_module
class RandomJit(object):
    """
    Randomly zoom in
    """
    def __init__(self, jit_upper_bound):
        self.jit_upper_bound = jit_upper_bound

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        # Deal with labels
        # 'alpha', 'bbox_b', 'bbox_l', 'bbox_r', 'bbox_t', 'h', 'l', 'occluded', 'ry', 'score', 'truncated', 'type', 'w', 'x', 'y', 'z'
        if labels:
            if isinstance(labels, list):
                # 2d bbox label will be take care at in visualDet3D/data/kitti/dataset/mono_dataset.py
                # it reproject 3d bbox back it image plane to get 2d bbox.
                # it's no need to jusify alpha as well, because mono_dataset.py do that when reproject option is on
                for obj in labels:
                    box_width = obj.bbox_r - obj.bbox_l
                    box_hight = obj.bbox_b - obj.bbox_t
                    dx = box_width * random_pkg.uniform(-self.jit_upper_bound, self.jit_upper_bound)
                    dy = box_hight * random_pkg.uniform(-self.jit_upper_bound, self.jit_upper_bound)
                    obj.bbox_r += dx
                    obj.bbox_l += dx
                    obj.bbox_b += dy
                    obj.bbox_t += dy
        
        return left_image, right_image, p2, p3, labels, image_gt, lidar

# Added by spiderkiller
@AUGMENTATION_DICT.register_module
class CutOut(object):
    """
    Randomly Cutout 2/4 hole in input image
    """
    def __init__(self, num_hole, mask_width):
        self.num_hole   = num_hole # How many hole add in the image
        self.mask_width = mask_width # How width the hole are
    
    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        
        h, w, c = left_image.shape
        
        for _ in range(self.num_hole):
            mx, my = ( random_pkg.randint(0, w-self.mask_width), 
                       random_pkg.randint(0, h-self.mask_width) ) # Left-top corner of the mask
        
            # Deal with Left_image
            left_image [my : my+self.mask_width, mx : mx+self.mask_width] = 0

            # Deal with right_image
            if right_image != None:
                right_image[my : my+self.mask_width, mx : mx+self.mask_width] = 0
        
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class ConvertToFloat(object):
    """
    Converts image data type to float.
    """
    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        return left_image.astype(np.float32), right_image if right_image is None else right_image.astype(np.float32), p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class Normalize(object):
    """
    Normalize the image
    """
    def __init__(self, mean, stds):
        self.mean = np.array(mean, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        left_image = left_image.astype(np.float32)
        left_image /= 255.0
        left_image -= np.tile(self.mean, int(left_image.shape[2]/self.mean.shape[0]))
        left_image /= np.tile(self.stds, int(left_image.shape[2]/self.stds.shape[0]))
        left_image.astype(np.float32)
        if right_image is not None:
            right_image = right_image.astype(np.float32)
            right_image /= 255.0
            right_image -= np.tile(self.mean, int(right_image.shape[2]/self.mean.shape[0]))
            right_image /= np.tile(self.stds, int(right_image.shape[2]/self.stds.shape[0]))
            right_image = right_image.astype(np.float32)
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class Resize(object):
    """
    Resize the image according to the target size height and the image height.
    If the image needs to be cropped after the resize, we crop it to self.size,
    otherwise we pad it with zeros along the right edge

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, size, preserve_aspect_ratio=True):
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        # print(f"self.preserve_aspect_ratio = {self.preserve_aspect_ratio}") # True
        if self.preserve_aspect_ratio:
            scale_factor = self.size[0] / left_image.shape[0]

            h = np.round(left_image.shape[0] * scale_factor).astype(int)
            w = np.round(left_image.shape[1] * scale_factor).astype(int)
            
            scale_factor_yx = (scale_factor, scale_factor)
        else:
            scale_factor_yx = (self.size[0] / left_image.shape[0], self.size[1] / left_image.shape[1])

            h = self.size[0]
            w = self.size[1]

        # resize
        left_image = cv2.resize(left_image, (w, h))
        if right_image is not None:
            right_image = cv2.resize(right_image, (w, h))
        if image_gt is not None:
            image_gt = cv2.resize(image_gt, (w, h), cv2.INTER_NEAREST)

        if len(self.size) > 1:

            # crop in
            if left_image.shape[1] > self.size[1]:
                left_image = left_image[:, 0:self.size[1], :]
                if right_image is not None:
                    right_image = right_image[:, 0:self.size[1], :]
                if image_gt is not None:
                    image_gt = image_gt[:, 0:self.size[1]]

            # pad out
            elif left_image.shape[1] < self.size[1]:
                padW = self.size[1] - left_image.shape[1]
                left_image  = np.pad(left_image,  [(0, 0), (0, padW), (0, 0)], 'constant')
                if right_image is not None:
                    right_image = np.pad(right_image, [(0, 0), (0, padW), (0, 0)], 'constant')
                if image_gt is not None:
                    if len(image_gt.shape) == 2:
                        image_gt = np.pad(image_gt, [(0, 0), (0, padW)], 'constant')
                    else:
                        image_gt = np.pad(image_gt, [(0, 0), (0, padW), (0, 0)], 'constant')

        if p2 is not None:
            p2[0, :]   = p2[0, :] * scale_factor_yx[1]
            p2[1, :]   = p2[1, :] * scale_factor_yx[0]
        
        if p3 is not None:
            p3[0, :]   = p3[0, :] * scale_factor_yx[1]
            p3[1, :]   = p3[1, :] * scale_factor_yx[0]
        
        if labels:
            if isinstance(labels, list):
                for obj in labels:
                    obj.bbox_l *= scale_factor_yx[1]
                    obj.bbox_r *= scale_factor_yx[1]
                    obj.bbox_t *= scale_factor_yx[0]
                    obj.bbox_b *= scale_factor_yx[0]
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class ResizeToFx(object):
    """
    Resize the image so that the Fx is aligned to a preset value

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, Fx=721.5337, Fy=None):
        self.Fx = Fx
        self.Fy = Fy if Fy is not None else Fx

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if p2 is None:
            print("P2 is None in ResizeToFx, will return the original input")
            return left_image, right_image, p2, p3, labels, image_gt, lidar
        
        h0 = left_image.shape[0]
        w0 = left_image.shape[1]
        fx0 = p2[0, 0]
        fy0 = p2[1, 1]

        h1 = int(h0 * self.Fy / fy0)
        w1 = int(w0 * self.Fx / fx0)

        scale_factor_yx = (float(h1) / h0, float(w1) / w0)

        # resize
        left_image = cv2.resize(left_image, (w1, h1))
        if right_image is not None:
            right_image = cv2.resize(right_image, (w1, h1))
        if image_gt is not None:
            image_gt = cv2.resize(image_gt, (w1, h1), cv2.INTER_NEAREST)

        if p2 is not None:
            p2[0, :]   = p2[0, :] * scale_factor_yx[1]
            p2[1, :]   = p2[1, :] * scale_factor_yx[0]
        
        if p3 is not None:
            p3[0, :]   = p3[0, :] * scale_factor_yx[1]
            p3[1, :]   = p3[1, :] * scale_factor_yx[0]
        
        if labels:
            if isinstance(labels, list):
                for obj in labels:
                    obj.bbox_l *= scale_factor_yx[1]
                    obj.bbox_r *= scale_factor_yx[1]
                    obj.bbox_t *= scale_factor_yx[0]
                    obj.bbox_b *= scale_factor_yx[0]

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomSaturation(object):
    """
    Randomly adjust the saturation of an image given a lower and upper bound,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.distort_prob = distort_prob
        self.lower = lower
        self.upper = upper

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            ratio = random.uniform(self.lower, self.upper)
            left_image[:, :, 1] *= ratio
            if right_image is not None:
                right_image[:, :, 1] *= ratio

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class CropTop(object):
    def __init__(self, crop_top_index=None, output_height=None):
        if crop_top_index is None and output_height is None:
            print("Either crop_top_index or output_height should not be None, set crop_top_index=0 by default")
            crop_top_index = 0
        if crop_top_index is not None and output_height is not None:
            print("Neither crop_top_index or output_height is None, crop_top_index will take over")
        self.crop_top_index = crop_top_index
        self.output_height = output_height

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, width = left_image.shape[0:2]

        if self.crop_top_index is not None:
            h_out = height - self.crop_top_index
            upper = self.crop_top_index
        else:
            h_out = self.output_height
            upper = height - self.output_height
        lower = height

        left_image = left_image[upper:lower]
        if right_image is not None:
            right_image = right_image[upper:lower]
        if image_gt is not None:
            image_gt = image_gt[upper:lower]
        ## modify calibration matrix
        if p2 is not None:
            p2[1, 2] = p2[1, 2] - upper               # cy' = cy - dv
            p2[1, 3] = p2[1, 3] - upper * p2[2, 3] # ty' = ty - dv * tz

        if p3 is not None:
            p3[1, 2] = p3[1, 2] - upper               # cy' = cy - dv
            p3[1, 3] = p3[1, 3] - upper * p3[2, 3] # ty' = ty - dv * tz

        
        if labels is not None:
            if isinstance(labels, list):
                # scale all coordinates
                for obj in labels:
                    obj.bbox_b -= upper
                    obj.bbox_t -= upper

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class CropRight(object):
    def __init__(self, crop_right_index=None, output_width=None):
        if crop_right_index is None and output_width is None:
            print("Either crop_right_index or output_width should not be None, set crop_right_index=0 by default")
            crop_right_index = 0
        if crop_right_index is not None and output_width is not None:
            print("Neither crop_right_index or output_width is None, crop_right_index will take over")
        self.crop_right_index = crop_right_index
        self.output_width = output_width

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, width = left_image.shape[0:2]

        lefter = 0
        if self.crop_right_index is not None:
            w_out = width - self.crop_right_index
            righter = w_out
        else:
            w_out = self.output_width
            righter = w_out
        
        if righter > width:
            print("does not crop right since it is larger")
            return left_image, right_image, p2, p3, labels

        # crop left image
        left_image = left_image[:, lefter:righter, :]

        # crop right image if possible
        if right_image is not None:
            right_image = right_image[:, lefter:righter, :]

        if image_gt is not None:
            image_gt = image_gt[:, lefter:righter]

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class FilterObject(object):
    """
        Filtering out object completely outside of the box;
    """
    def __init__(self):
        pass

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, width = left_image.shape[0:2]

        if labels is not None:
            new_labels = []
            if isinstance(labels, list):
                # scale all coordinates
                for obj in labels:
                    is_outside = (
                        obj.bbox_b < 0 or obj.bbox_t > height or obj.bbox_r < 0 or obj.bbox_l > width
                    )
                    if not is_outside:
                        new_labels.append(obj)
        else:
            new_labels = None
        
        return left_image, right_image, p2, p3, new_labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomCropToWidth(object):
    def __init__(self, width:int):
        self.width = width

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        height, original_width = left_image.shape[0:2]

        

        if self.width > original_width:
            print("does not crop since it is larger")
            return left_image, right_image, p2, p3, labels, image_gt


        lefter = np.random.randint(0, original_width - self.width)
        righter = lefter + self.width
        # crop left image
        left_image = left_image[:, lefter:righter, :]

        # crop right image if possible
        if right_image is not None:
            right_image = right_image[:, lefter:righter, :]

        if image_gt is not None:
            image_gt = image_gt[:, lefter:righter]

        ## modify calibration matrix
        if p2 is not None:
            p2[0, 2] = p2[0, 2] - lefter               # cy' = cy - dv
            p2[0, 3] = p2[0, 3] - lefter * p2[2, 3] # ty' = ty - dv * tz

        if p3 is not None:
            p3[0, 2] = p3[0, 2] - lefter               # cy' = cy - dv
            p3[0, 3] = p3[0, 3] - lefter * p3[2, 3] # ty' = ty - dv * tz

        
        if labels:
            # scale all coordinates
            if isinstance(labels, list):
                for obj in labels:
                        
                    obj.bbox_l -= lefter
                    obj.bbox_r -= lefter

            

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomMirror(object):
    """
    Randomly mirror an image horzontially, given a mirror probabilty.

    Also, adjust all box cordinates accordingly.
    """
    def __init__(self, mirror_prob):
        self.mirror_prob = mirror_prob
        self.projector = BBox3dProjector()

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        _, width, _ = left_image.shape

        if random.rand() <= self.mirror_prob:

            left_image = left_image[:, ::-1, :]
            left_image = np.ascontiguousarray(left_image)

            if right_image is not None:
                right_image = right_image[:, ::-1, :]
                right_image = np.ascontiguousarray(right_image)

                left_image, right_image = right_image, left_image
            if image_gt is not None:
                image_gt = image_gt[:, ::-1]
                image_gt = np.ascontiguousarray(image_gt)

            # flip the coordinates w.r.t the horizontal flip (only adjust X)
            if p2 is not None and p3 is not None:
                p2, p3 = p3, p2
            if p2 is not None:
                p2[0, 3] = -p2[0, 3]
                p2[0, 2] = left_image.shape[1] - p2[0, 2] - 1
            if p3 is not None:
                p3[0, 3] = -p3[0, 3]
                p3[0, 2] = left_image.shape[1] - p3[0, 2] - 1
            if labels:
                if isinstance(labels, list):
                    square_P2 = np.eye(4)
                    square_P2[0:3, :] = p2
                    p2_inv = np.linalg.inv(square_P2)
                    for obj in labels:
                        # In stereo horizontal 2D boxes will be fixed later when we use 3D projection as 2D anchor box
                        obj.bbox_l, obj.bbox_r = left_image.shape[1] - obj.bbox_r - 1, left_image.shape[1] - obj.bbox_l - 1
                        
                        # 3D centers
                        z = obj.z
                        obj.x = -obj.x

                        # yaw
                        ry = obj.ry
                        ry = (-pi - ry) if ry < 0 else (pi - ry)
                        while ry > pi: ry -= pi * 2
                        while ry < (-pi): ry += pi * 2
                        obj.ry = ry

                        # alpha 
                        obj.alpha = theta2alpha_3d(ry, obj.x, z, p2)
                
            if lidar is not None:
                lidar[:, :, 0] = -lidar[:, :, 0]
        
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomWarpAffine(object):
    """
        Randomly random scale and random shift the image. Then resize to a fixed output size. 
    """
    def __init__(self, scale_lower=0.6, scale_upper=1.4, shift_border=128, output_w=1280, output_h=384):
        self.scale_lower    = scale_lower
        self.scale_upper    = scale_upper
        self.shift_border   = shift_border
        self.output_w       = output_w
        self.output_h       = output_h

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        s_original = max(left_image.shape[0], left_image.shape[1])
        center_original = np.array([left_image.shape[1] / 2., left_image.shape[0] / 2.], dtype=np.float32)
        scale = s_original * np.random.uniform(self.scale_lower, self.scale_upper)
        center_w = np.random.randint(low=self.shift_border, high=left_image.shape[1] - self.shift_border)
        center_h = np.random.randint(low=self.shift_border, high=left_image.shape[0] - self.shift_border)

        final_scale = max(self.output_w, self.output_h) / scale
        final_shift_w = self.output_w / 2 - center_w * final_scale
        final_shift_h = self.output_h / 2 - center_h * final_scale
        affine_transform = np.array(
            [
                [final_scale, 0, final_shift_w],
                [0, final_scale, final_shift_h]
            ], dtype=np.float32
        )

        left_image = cv2.warpAffine(left_image, affine_transform,
                                    (self.output_w, self.output_h), flags=cv2.INTER_LINEAR)
        if right_image is not None:
            right_image = cv2.warpAffine(right_image, affine_transform,
                                    (self.output_w, self.output_h), flags=cv2.INTER_LINEAR)

        if image_gt is not None:
            image_gt = cv2.warpAffine(image_gt, affine_transform,
                                    (self.output_w, self.output_h), flags=cv2.INTER_LINEAR)

        if p2 is not None:
            p2[0:2, :] *= final_scale
            p2[0, 2] = p2[0, 2] + final_shift_w               # cy' = cy - dv
            p2[0, 3] = p2[0, 3] + final_shift_w * p2[2, 3] # ty' = ty - dv * tz
            p2[1, 2] = p2[1, 2] + final_shift_h               # cy' = cy - dv
            p2[1, 3] = p2[1, 3] + final_shift_h * p2[2, 3] # ty' = ty - dv * tz

        if p3 is not None:
            p3[0:2, :] *= final_scale
            p3[0, 2] = p3[0, 2] + final_shift_w               # cy' = cy - dv
            p3[0, 3] = p3[0, 3] + final_shift_w * p3[2, 3] # ty' = ty - dv * tz
            p3[1, 2] = p3[1, 2] + final_shift_h               # cy' = cy - dv
            p3[1, 3] = p3[1, 3] + final_shift_h * p3[2, 3] # ty' = ty - dv * tz
        
        if labels:
            if isinstance(labels, list):
                for obj in labels:
                    obj.bbox_l = obj.bbox_l * final_scale + final_shift_w
                    obj.bbox_r = obj.bbox_r * final_scale + final_shift_w
                    obj.bbox_t = obj.bbox_t * final_scale + final_shift_h
                    obj.bbox_b = obj.bbox_b * final_scale + final_shift_h

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomHue(object):
    """
    Randomly adjust the hue of an image given a delta degree to rotate by,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            shift = random.uniform(-self.delta, self.delta)
            left_image[:, :, 0] += shift
            left_image[:, :, 0][left_image[:, :, 0] > 360.0] -= 360.0
            left_image[:, :, 0][left_image[:, :, 0] < 0.0] += 360.0
            if right_image is not None:
                right_image[:, :, 0] += shift
                right_image[:, :, 0][right_image[:, :, 0] > 360.0] -= 360.0
                right_image[:, :, 0][right_image[:, :, 0] < 0.0] += 360.0
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class ConvertColor(object):
    """
    Converts color spaces to/from HSV and RGB
    """
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        # RGB --> HSV
        if self.current == 'RGB' and self.transform == 'HSV':
            left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2HSV)
            if right_image is not None:
                right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2HSV)

        # HSV --> RGB
        elif self.current == 'HSV' and self.transform == 'RGB':
            left_image = cv2.cvtColor(left_image, cv2.COLOR_HSV2RGB)
            if right_image is not None:
                right_image = cv2.cvtColor(right_image, cv2.COLOR_HSV2RGB)

        else:
            raise NotImplementedError

        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomContrast(object):
    """
    Randomly adjust contrast of an image given lower and upper bound,
    and a distortion probability.
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.lower = lower
        self.upper = upper
        self.distort_prob = distort_prob

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            alpha = random.uniform(self.lower, self.upper)
            left_image *= alpha
            if right_image is not None:
                right_image *= alpha
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomBrightness(object):
    """
    Randomly adjust the brightness of an image given given a +- delta range,
    and a distortion probability.
    """
    def __init__(self, distort_prob, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            delta = random.uniform(-self.delta, self.delta)
            left_image += delta
            if right_image is not None:
                right_image += delta
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class RandomEigenvalueNoise(object):
    """
        Randomly apply noise in RGB color channels based on the eigenvalue and eigenvector of ImageNet
    """
    def __init__(self, distort_prob=1.0,
                       alphastd=0.1,
                       eigen_value=np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32),
                       eigen_vector=np.array([
                            [-0.58752847, -0.69563484, 0.41340352],
                            [-0.5832747, 0.00994535, -0.81221408],
                            [-0.56089297, 0.71832671, 0.41158938]
                        ], dtype=np.float32)
                ):
        self.distort_prob = distort_prob
        self._eig_val = eigen_value
        self._eig_vec = eigen_vector
        self.alphastd = alphastd

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):
        if random.rand() <= self.distort_prob:
            alpha = np.random.normal(scale=self.alphastd, size=(3, ))
            noise = np.dot(self._eig_vec, self._eig_val * alpha) * 255

            left_image += noise
            if right_image is not None:
                right_image += noise
            
        return left_image, right_image, p2, p3, labels, image_gt, lidar

@AUGMENTATION_DICT.register_module
class PhotometricDistort(object):
    """
    Jitter contrast, saturation and hue
    Packages all photometric distortions into a single transform.
    """
    def __init__(self, distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32):

        self.p = distort_prob
        self.contrast_range = (contrast_lower, contrast_upper)
        self.saturation_range = (saturation_lower, saturation_upper)
        self.hue_delta = hue_delta
        self.brightness_delta = brightness_delta

    def __call__(self, left_image, right_image=None, p2=None, p3=None, labels=None, image_gt=None, lidar=None):

        aug_list = [
            edict(type_name='RandomBrightness', keywords=edict(distort_prob = self.p, delta = self.brightness_delta)),
            edict(type_name='RandomContrast'  , keywords=edict(distort_prob = self.p, lower = self.contrast_range[0], upper = self.contrast_range[1])),
            edict(type_name='ConvertColor'    , keywords=edict(transform='HSV')),
            edict(type_name='RandomSaturation', keywords=edict(distort_prob = self.p, lower = self.saturation_range[0], upper = self.saturation_range[1])),
            edict(type_name='RandomHue'       , keywords=edict(distort_prob = self.p, delta = self.hue_delta)),
            edict(type_name='ConvertColor'    , keywords=edict(current = 'HSV', transform = 'RGB')),
        ]

        if random.rand() <= 0.5:
            # do contrast last
            random_contrast = aug_list.pop(1)
            aug_list.append(random_contrast)
        
        # compose transformation
        distortion = AugmentataionComposer(aug_list, is_return_all = True)
        return distortion(left_image.copy(), right_image if right_image is None else right_image.copy(), p2, p3, labels, image_gt, lidar)
