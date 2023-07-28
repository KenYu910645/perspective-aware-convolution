import cv2
import os 
import glob
import random 
import shutil
from iou_3d import get_3d_box, box3d_iou, box2d_iou
import json 
import numpy as np 

random.seed(5278)

VEHICLES = ["Car"] # What kind of object added to base image
N_DATASET_REPEAT = 1 # The total size of output image = N_DATASET_REPEAT * 7481
N_ADD_OBJ = 3 # How many newly added object in a image
SOLID_RATIO = 1.0 # How solid will the added patch looks like
IS_SEG_GT = True # Use segmentation label or not

IMAGE_DIR = "/home/lab530/KenYu/kitti/training/image_2/"
LABEL_DIR = "/home/lab530/KenYu/kitti/training/label_2/"
CALIB_DIR = "/home/lab530/KenYu/kitti/training/calib/"
SEGMT_DIR = "/home/lab530/KenYu/kitti/training/image_bbox_label/" # Only used when IS_SEG_GT = true
TRAIN_SPLIT_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/train.txt"
VALID_SPLIT_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt"
OUT_IMAGE_DIR = f"/home/lab530/KenYu/kitti_seg_{N_ADD_OBJ}/training/image_2/"
OUT_LABEL_DIR = f"/home/lab530/KenYu/kitti_seg_{N_ADD_OBJ}/training/label_2/"
OUT_CALIB_DIR = f"/home/lab530/KenYu/kitti_seg_{N_ADD_OBJ}/training/calib/"
OUT_SPLIT_DIR = f"/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/kitti_seg_{N_ADD_OBJ}_split/"

# TODO 2D IOU check might make it a good way to do it ? 
# TODO multiple object generation should render object by the z_3d order, 
#      but if we use half transparant rander, this won't be a big problem

class Object:
    def __init__(self, str_line, idx_img = None, idx_line = None):
        # str_line should be 'Car 0.00 0 -1.58 587.19 178.91 603.38 191.75 1.26 1.60 3.56 -1.53 1.89 73.44 -1.60'
        self.idx_img  = idx_img # this obj belong to which image 
        self.idx_line = idx_line # this obj belong to which line in label.txt
        self.raw_str = str_line
        sl = str_line.split()
        self.category, self.truncated, self.occluded, self.alpha = sl[0], float(sl[1]), int(sl[2]), float(sl[3])
        self.x_min, self.y_min, self.x_max, self.y_max = [int(float(i)) for i in sl[4:8]]
        self.h, self.w, self.l = [float(i) for i in sl[8:11]]
        self.x_3d, self.y_3d, self.z_3d = [float(i) for i in sl[11:14]]
        self.rot_y = float(sl[14])
        self.area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        self.corners = get_3d_box((self.l, self.w, self.h),
                                   self.rot_y,
                                  (self.x_3d, self.y_3d, self.z_3d))
        self.seg_points = [] # Only use if IS_SEG_GT == true

        # Find segment labels
        json_path = SEGMT_DIR + f"{idx_img}_{idx_line}.json"
        
        if IS_SEG_GT and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                raw_json = json.load(f)
                self.seg_points = raw_json['shapes'][0]["points"]
        
# Clean output directory 
print("Clean output directory : " + str(OUT_IMAGE_DIR))
print("Clean output directory : " + str(OUT_LABEL_DIR))
print("Clean output directory : " + str(OUT_CALIB_DIR))
print("Clean output directory : " + str(OUT_SPLIT_DIR))
shutil.rmtree(OUT_IMAGE_DIR, ignore_errors=True)
shutil.rmtree(OUT_LABEL_DIR, ignore_errors=True)
shutil.rmtree(OUT_CALIB_DIR, ignore_errors=True)
shutil.rmtree(OUT_SPLIT_DIR, ignore_errors=True)
os.mkdir(OUT_IMAGE_DIR)
os.mkdir(OUT_LABEL_DIR)
os.mkdir(OUT_CALIB_DIR)
os.mkdir(OUT_SPLIT_DIR)

# Record image that can't augumented 
tartar = []

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
    shutil.copyfile(IMAGE_DIR + l + ".png", OUT_IMAGE_DIR + l + ".png")
    shutil.copyfile(LABEL_DIR + l + ".txt", OUT_LABEL_DIR + l + ".txt")
    shutil.copyfile(CALIB_DIR + l + ".txt", OUT_CALIB_DIR + l + ".txt")

# Output split file. To specify how to split train and valid data
with open(OUT_SPLIT_DIR + 'train.txt', 'w') as f:
    for idx_repeat in range(N_DATASET_REPEAT):
        [f.write(f"{idx_repeat}{i[1:]}\n") for i in img_idxs]

shutil.copyfile(VALID_SPLIT_TXT, OUT_SPLIT_DIR + "val.txt")
print(f"Ouptut train.txt to {OUT_SPLIT_DIR}")
print(f"Ouptut val.txt to {OUT_SPLIT_DIR}")

# Find ground trues object that are appropiate
gts_add = []
for idx_add in [i.split('/')[-1].split('.')[0] for i in img_paths]:
    # Get add_object
    with open(LABEL_DIR + f"{idx_add}.txt") as f:
        gt_lines = f.read().splitlines()
        gt_lines = list(gt_lines for gt_lines in gt_lines if gt_lines) # Delete empty lines
        gt_add = [Object(gt, idx_add, idx_line) for idx_line, gt in enumerate(gt_lines)]
    
    # Filter inappropiate objs
    for gt in gt_add:
        if gt.category in VEHICLES and gt.truncated < 0.5 and gt.occluded == 0.0 and gt.area > 3000:
            if IS_SEG_GT and len(gt.seg_points) == 0 : continue
            gts_add.append(gt)

print(f"Total number of object that is Ok to add : {len(gts_add)}")

for idx_repeat in range(N_DATASET_REPEAT):
    for idx_src, img_path in zip(img_idxs, img_paths):
        # Skip tartar images, This might cause label.txt and calib.txt have a problem
        # if idx_src in tartar: continue 
        
        # Load image
        img_src = cv2.imread(img_path)

        # Load source gt objects
        with open(LABEL_DIR + f"{idx_src}.txt") as f:
            gt_lines = f.read().splitlines()
            gt_lines = list(gt_lines for gt_lines in gt_lines if gt_lines) # Delete empty lines
            gts_src = [Object(gt) for gt in gt_lines]

        # Add N_ADD_OBJ objects in the source image
        for obj_idx in range(N_ADD_OBJ):
            random.shuffle(gts_add)

            # Avoid 3D bbox collide with each other on BEV
            gt_rst = None # Result
            for gt_add in gts_add:
                # Avoid using object without labels
                if IS_SEG_GT and len(gt_add.seg_points) == 0: continue

                is_good_spawn = True

                for gt_src in gts_src:
                    # Get 2D IOU
                    iou_2d = box2d_iou((gt_add.x_min, gt_add.y_min, gt_add.x_max, gt_add.y_max),
                                        (gt_src.x_min, gt_src.y_min, gt_src.x_max, gt_src.y_max))
                    # Get 3D IOU
                    try: # TODO this is because I can't solve ZeroDivision in iou_3d.py
                        iou_3db, iou_bev = box3d_iou(gt_src.corners, gt_add.corners)
                    except: 
                        iou_3db, iou_bev = (0, 0)
                        is_good_spawn = False

                    # Avoid far object paste on nearer object
                    if iou_2d > 0.0 and gt_add.z_3d > gt_src.z_3d: is_good_spawn = False

                    # Avoid 3D bbox collide with each other on BEV
                    if iou_bev > 0.0: is_good_spawn = False
                    #
                    if not is_good_spawn: break
                
                if is_good_spawn: gt_rst = gt_add
                # 
                if gt_rst != None: break

            # If couldn't find any augmented image, copy paste image without modification
            if gt_rst == None:
                tartar.append(idx_src)
                print(f"[WARNING] Cannot find any suitable gt to add to {idx_src}.png")
                break

            # Get aug image
            img_add = cv2.imread(IMAGE_DIR + f"{gt_rst.idx_img}.png")
            
            h_src, w_src, _ = img_src.shape
            h_add, w_add, _ = img_add.shape

            if img_src.shape != img_add.shape:
                img_add = cv2.resize(img_add, (img_src.shape[1], img_src.shape[0]))

            # 2D boudning box Saturation. 
            gt_rst.x_max = min(gt_rst.x_max, w_add-1)
            gt_rst.y_max = min(gt_rst.y_max, h_add-1)
            
            # if img_src and img_add has the same resolution
            xmin = int(gt_rst.x_min * (w_src/w_add))
            xmax = int(gt_rst.x_max * (w_src/w_add))
            ymin = int(gt_rst.y_min * (h_src/h_add))
            ymax = int(gt_rst.y_max * (h_src/h_add))

            #
            roi_add = cv2.resize(img_add[gt_rst.y_min:gt_rst.y_max, gt_rst.x_min:gt_rst.x_max], 
                                (xmax - xmin, ymax - ymin),
                                interpolation=cv2.INTER_AREA)

            if IS_SEG_GT:
                mask = np.zeros_like(img_src[:, :, 1])
                
                obj_countor = []
                for xp, yp in gt_rst.seg_points:
                    xp = int((xp + gt_rst.x_min) * (w_src/w_add))
                    yp = int((yp + gt_rst.y_min) * (h_src/h_add))
                    obj_countor.append(np.array([[xp, yp]], dtype=np.int32))

                cv2.drawContours(mask, np.array([obj_countor]), -1, 255, -1)
                img_src[mask == 255] = img_add[mask == 255]
            else:
                # Add new object on augmented image and .txt
                img_src[ymin:ymax, xmin:xmax] = SOLID_RATIO*roi_add + (1-SOLID_RATIO)*img_src[ymin:ymax, xmin:xmax]
            # Add new object to source gts
            gts_src.append(gt_rst)
        
        # Output calib.txt, directly copy source
        shutil.copyfile(CALIB_DIR + f"{idx_src}.txt", OUT_CALIB_DIR + f"{idx_repeat}{idx_src[1:]}.txt")

        # Output label.txt
        with open(OUT_LABEL_DIR + f"{idx_repeat}{idx_src[1:]}.txt", 'w') as f:
            for gt in gts_src: f.write(gt.raw_str + '\n')
        
        # Output image.png
        cv2.imwrite(OUT_IMAGE_DIR + f"{idx_repeat}{idx_src[1:]}.png", img_src)
        # 
        print(f"Processed {idx_src}/{len(img_paths)} image of {idx_repeat+1}/{N_DATASET_REPEAT} repeats")

print(f"tartar = {tartar}")
print(f"There are {len(tartar)} images can't be augumented!")