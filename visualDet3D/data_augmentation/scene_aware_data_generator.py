import matplotlib # Disable GUI
matplotlib.use('Agg')

import cv2
import os 
import random 
import shutil
import copy
import json 
import numpy as np 
from math import sqrt
import argparse

from visualDet3D.utils.iou_3d import get_3d_box, box3d_iou, box2d_iou, box2d_iog
from visualDet3D.utils.util_kitti import kitti_calib_file_parser, KITTI_Object

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_dataset_repeat', type=int,   default=5, help='The total size of output image = N_DATASET_REPEAT * 7481')
parser.add_argument('--num_add_obj'       , type=int,   default=3, help='How many newly added object in a image')
parser.add_argument('--solid_ratio'       , type=float, default=1.0, help='How solid will the added patch looks like')
parser.add_argument('--use_segement'   , action='store_true', help='whether to use segment data')
parser.add_argument('--use_scene_aware', action='store_true', help='whether to use scene-aware data')
parser.add_argument('--use_z_jitter'   , action='store_true', help='whether to use z-axis jittering')

args = parser.parse_args()

print(args)

# random.seed(5278)

VEHICLES = ["Car"] # What kind of object added to base image
N_DATASET_REPEAT = args.num_dataset_repeat
N_ADD_OBJ        = args.num_add_obj
SOLID_RATIO      = args.solid_ratio
IS_SEG_GT        = args.use_segement
IS_DEP_CHECK     = args.use_scene_aware
IS_OBJ_Z_CHANGE  = args.use_z_jitter

IMAGE_DIR = "/home/lab530/KenYu/kitti/training/image_2/"
LABEL_DIR = "/home/lab530/KenYu/kitti/training/label_2/"
CALIB_DIR = "/home/lab530/KenYu/kitti/training/calib/"
SEGMT_DIR = "/home/lab530/KenYu/kitti/training/image_bbox_label/" # Only used when IS_SEG_GT = true
DEPTH_DIR = "/home/lab530/KenYu/kitti/training/image_depth/" # Only used when IS_DEP_CHECK = true
TRAIN_SPLIT_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/train.txt"
VALID_SPLIT_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt"

# Get experiment name
exp_name = "kitti"
exp_name += "_seg" if IS_SEG_GT else "_box"
exp_name += f"_solid_{int(SOLID_RATIO*10)}"
exp_name += f"_obj_{N_ADD_OBJ}"
exp_name += "_zJitter" if IS_OBJ_Z_CHANGE else ""
exp_name += "_sceneAware" if IS_DEP_CHECK else ""

OUT_EXPRI_DIR = f"/home/lab530/KenYu/{exp_name}/"
OUT_IMAGE_DIR = os.path.join(OUT_EXPRI_DIR, "training", "image_2")
OUT_LABEL_DIR = os.path.join(OUT_EXPRI_DIR, "training", "label_2")
OUT_CALIB_DIR = os.path.join(OUT_EXPRI_DIR, "training", "calib")
OUT_SPLIT_DIR = f"/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/{exp_name}_split/"

class Object(KITTI_Object):
    def __init__(self, str_line, tf_matrix, idx_img = None, idx_line = None):
        super().__init__(str_line, tf_matrix, idx_img, idx_line)
        self.corners_3d = get_3d_box((self.l, self.w, self.h), self.rot_y, (self.x3d, self.y3d, self.z3d))
        
        # Find segment labels
        json_path = SEGMT_DIR + f"{idx_img}_{idx_line}.json"
        #
        self.seg_points = [] # Only use if IS_SEG_GT == true
        if IS_SEG_GT and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                raw_json = json.load(f)
                self.seg_points = raw_json['shapes'][0]["points"]

    def reprojection(self):
        super().reprojection()
        self.corners_3d = get_3d_box((self.l, self.w, self.h), self.rot_y, (self.x3d, self.y3d, self.z3d))

def check_depth_map(gt_add, tar_depth):
    try:
        # Calculate the number of pixels above the threshold
        num_above_threshold = np.count_nonzero(tar_depth[gt_add.ymin:gt_add.ymax, gt_add.xmin:gt_add.xmax] < gt_add.cz)
        percent_above_threshold = num_above_threshold / gt_add.area
        # print(f"percent_above_threshold = {percent_above_threshold}")
        
        if percent_above_threshold > 0.25: 
            return False
        else: 
            return True
    except TypeError:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def check_suitable(gt_add, gts_tar):
    '''
    Check whether gt_add is sutiable to add to gts_tar
    '''
    for gt_tar in gts_tar:
        # Get 2D IOU, Avoid far object paste on nearer object
        iou_2d = box2d_iou((gt_add.xmin, gt_add.ymin, gt_add.xmax, gt_add.ymax),
                           (gt_tar.xmin, gt_tar.ymin, gt_tar.xmax, gt_tar.ymax))
        if iou_2d > 0.0 and gt_add.z3d > gt_tar.z3d: return False
        
        # IoG check
        b1_iog, b2_iog = box2d_iog((gt_add.xmin, gt_add.ymin, gt_add.xmax, gt_add.ymax),
                                   (gt_tar.xmin, gt_tar.ymin, gt_tar.xmax, gt_tar.ymax))
        if max(b1_iog, b2_iog) > 0.7 : return False
        
        # Get 3D IOU, Avoid 3D bbox collide with each other on BEV
        try: # TODO this is because I can't solve ZeroDivision in iou_3d.py
            iou_3db, iou_bev = box3d_iou(gt_tar.corners_3d, gt_add.corners_3d)
        except Exception as e:
            print("Error:", str(e))
            iou_3db, iou_bev = (0, 0)
            return False
        if iou_bev > 0.0: return False
    return True

# Clean output directory 
print("Clean output directory : " + str(OUT_EXPRI_DIR))
print("Clean output directory : " + str(OUT_SPLIT_DIR))
shutil.rmtree(OUT_EXPRI_DIR, ignore_errors=True)
shutil.rmtree(OUT_SPLIT_DIR, ignore_errors=True)
os.mkdir(OUT_EXPRI_DIR)
os.mkdir(os.path.join(OUT_EXPRI_DIR, "training"))
os.mkdir(OUT_LABEL_DIR)
os.mkdir(OUT_CALIB_DIR)
os.mkdir(OUT_IMAGE_DIR)
os.mkdir(OUT_SPLIT_DIR)

# Only augumented training split, don't augment validation set
img_paths = []
with open(TRAIN_SPLIT_TXT, 'r') as f:
    lines = f.read().splitlines()
    img_idxs = list(lines for lines in lines if lines) # Delete empty lines
img_paths = [IMAGE_DIR + l + '.png' for l in lines]
print(f"Find {len(img_paths)} source images.")

# Copy validation split to destination, don't augument
with open(VALID_SPLIT_TXT, 'r') as f:
    lines = f.read().splitlines()
    lines = list(lines for lines in lines if lines) # Delete empty lines
for l in lines:
    shutil.copyfile(os.path.join(IMAGE_DIR, f"{l}.png"), os.path.join(OUT_IMAGE_DIR, f"{l}.png"))
    shutil.copyfile(os.path.join(LABEL_DIR, f"{l}.txt"), os.path.join(OUT_LABEL_DIR, f"{l}.txt"))
    shutil.copyfile(os.path.join(CALIB_DIR, f"{l}.txt"), os.path.join(OUT_CALIB_DIR, f"{l}.txt"))

# Output split file. To specify how to split train and valid data
with open(OUT_SPLIT_DIR + 'train.txt', 'w') as f:
    for idx_repeat in range(N_DATASET_REPEAT):
        [f.write(f"{idx_repeat}{i[1:]}\n") for i in img_idxs]

shutil.copyfile(VALID_SPLIT_TXT, os.path.join(OUT_SPLIT_DIR, "val.txt"))
print(f"Ouptut train.txt to {OUT_SPLIT_DIR}")
print(f"Ouptut val.txt to {OUT_SPLIT_DIR}")


#########################
### Get Instance Pool ###
#########################
# Only Use instance in training set
objs_pool = []
with open(TRAIN_SPLIT_TXT, 'r') as f:
    lines    = f.read().splitlines()
    images_fn = list(lines for lines in lines if lines) # Delete empty lines
print(f"Find {len(images_fn)} source images in training set")

for img_fn in images_fn:
    P2   = kitti_calib_file_parser(os.path.join(CALIB_DIR, f"{img_fn}.txt"))
    with open(os.path.join(LABEL_DIR, f"{img_fn}.txt")) as f:
        lines = f.read().splitlines()
        lines = list(lines for lines in lines if lines) # Delete empty lines
    objs =  [Object(str_line + " NA",
                    idx_img = img_fn,
                    idx_line = idx_line, 
                    tf_matrix = P2) for idx_line, str_line in enumerate(lines)]
    #
    for obj in objs:
        # Filter inappropiate objs
        if obj.category in VEHICLES and obj.truncated < 0.5 and obj.occluded == 0.0 and obj.area > 3000:
            if IS_SEG_GT and len(obj.seg_points) == 0 : continue
            objs_pool.append(obj)
print(f"Number of object in instance pool: {len(objs_pool)}")


###############################
### Copy-Paste Augmentation ###
###############################
aug_fail_img = []
for idx_repeat in range(N_DATASET_REPEAT):
    for i_img, img_name in enumerate(img_idxs):
        
        # Load image
        img_tar = cv2.imread(os.path.join(IMAGE_DIR, f"{img_name}.png"))

        # Load Calibration
        P2 = kitti_calib_file_parser(CALIB_DIR + f"{img_name}.txt")
        
        # Load Depth Map
        depth_map_path = os.path.join(DEPTH_DIR, f"{img_name}.png")
        img_tar_depth = None
        if IS_DEP_CHECK:
            if os.path.exists(depth_map_path):
                img_tar_depth = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"[WARNING] No depth map find in {depth_map_path}")
        
        # Load Ground true object in target image
        with open(os.path.join(LABEL_DIR, f"{img_name}.txt"), 'r') as f:
            lines = f.read().splitlines()
            lines = list(lines for lines in lines if lines) # Delete empty lines
        gts_tar =  [Object(str_line + " NA",
                        idx_img = os.path.join(LABEL_DIR, f"{img_name}.txt").split('/')[-1].split('.')[0],
                        idx_line = idx_line, 
                        tf_matrix = P2) for idx_line, str_line in enumerate(lines)]
        
        # Add objects in the target image
        for obj_idx in range(N_ADD_OBJ):
            random.shuffle(objs_pool)

            gt_rst_new = None
            gt_rst_old = None
            for i_gt_add, gt_add in enumerate(objs_pool):
                # Don't use object without segmented label
                if IS_SEG_GT and len(gt_add.seg_points) == 0: continue
                #
                gt_add_old = copy.deepcopy( gt_add )
                gt_add_new = copy.deepcopy( gt_add )
                
                # Jitter Z direction
                if IS_OBJ_Z_CHANGE:
                    cz_new = random.uniform( gt_add.cz*0.8, gt_add.cz*1.2 )
                    s = gt_add.cz / cz_new
                    
                    d_bev_old = sqrt(gt_add.x3d**2 + gt_add.z3d**2)
                    d_bev_new = sqrt(cz_new**2 - (gt_add.y3d - gt_add.h/2)**2)
                    
                    gt_add_new.x3d *= d_bev_new / d_bev_old
                    gt_add_new.z3d *= d_bev_new / d_bev_old
                    gt_add_new.P2 = P2
                    gt_add_new.reprojection()
                
                # Check whether it's a good spawn by scanning through all the existed groundTrue
                if not check_suitable(gt_add_new, gts_tar): continue
                
                # Get aug image
                img_src = cv2.imread(IMAGE_DIR + f"{gt_add_new.idx_img}.png")
                
                img_tar_h, img_tar_w, _ = img_tar.shape
                img_src_h, img_src_w, _ = img_src.shape
                
                # Source image
                patch_src = img_src[gt_add_old.ymin:gt_add_old.ymax,
                                    gt_add_old.xmin:gt_add_old.xmax]
                
                patch_tar_w, patch_tar_h = (int(gt_add_new.xmax - gt_add_new.xmin),
                                            int(gt_add_new.ymax - gt_add_new.ymin))

                # Resize source patch to fit the target patch
                patch_src = cv2.resize(patch_src, (patch_tar_w, patch_tar_h), interpolation=cv2.INTER_AREA)
                if IS_SEG_GT:
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
                if IS_DEP_CHECK:
                    if os.path.exists(depth_map_path):
                        if not check_depth_map(gt_add_new, img_tar_depth): continue
                    else:
                        print(f"[WARNING] Skip depth map check because no depth map found in {depth_map_path}")

                gt_rst_old = gt_add_old
                gt_rst_new = gt_add_new
                break

            # If couldn't find any augmented image, copy paste image without modification
            if gt_rst_new == None:
                aug_fail_img.append(img_name)
                print(f"[WARNING] Cannot find suitable gt to add to {img_name}.png")
                break
            
            # Paste the source instance on target image
            if IS_SEG_GT:
                mask = np.zeros_like(patch_src[:, :, 1])
                
                obj_countor = []
                for xp, yp in gt_rst_new.seg_points:
                    obj_countor.append(np.array([[int(xp), int(yp)]], dtype=np.int32))

                cv2.drawContours(mask, np.array([obj_countor]), -1, 255, -1)
                img_tar[gt_rst_new.ymin:gt_rst_new.ymax, gt_rst_new.xmin:gt_rst_new.xmax][mask == 255] = patch_src[mask == 255]
            else:
                # Add new object on augmented image and .txt
                img_tar[gt_rst_new.ymin:gt_rst_new.ymax, gt_rst_new.xmin:gt_rst_new.xmax] = SOLID_RATIO*patch_src + \
                img_tar[gt_rst_new.ymin:gt_rst_new.ymax, gt_rst_new.xmin:gt_rst_new.xmax]*(1-SOLID_RATIO)
            
            # Add new object's label to source gts
            gts_tar.append(gt_rst_new)
        
        # Output calib.txt, directly copy source
        shutil.copyfile(os.path.join(CALIB_DIR    , f"{img_name}.txt"), 
                        os.path.join(OUT_CALIB_DIR, f"{idx_repeat}{img_name[1:]}.txt"))

        # Output label.txt
        # print(os.path.join(OUT_LABEL_DIR, f"{idx_repeat}{img_name[1:]}.txt"))
        with open(os.path.join(OUT_LABEL_DIR, f"{idx_repeat}{img_name[1:]}.txt"), 'w') as f:
            for gt in gts_tar: f.write(gt.__str__() + '\n')
        
        # Output image.png
        cv2.imwrite( os.path.join(OUT_IMAGE_DIR, f"{idx_repeat}{img_name[1:]}.png") , img_tar)
        # 
        print(f"Processed {i_img}/{len(img_idxs)} image of {idx_repeat+1}/{N_DATASET_REPEAT} repeats")
    
print(f"aug_fail_img = {aug_fail_img}")
print(f"Can't find sutiable object to paste for {len(aug_fail_img)} times")
