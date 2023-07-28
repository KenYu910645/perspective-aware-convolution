'''
Convert KITTI label to yolo label format txt
'''

# YOLO label format:
# <object_class> <x> <y> <width> <height>
# (x,y) is the center of rectangle.
# All four value are [0,1], representing the ratio to image.shape
'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''
import pprint
import json
from shutil import rmtree
import os
import os.path as osp
import cv2

# TODO 
# Input directory (KITTI labels)
kitti_dir = '/home/spiderkiller/kitti_dataset/label_2/'
image_dir = '/home/spiderkiller/kitti_dataset/image_2/' # Need images because don't know resolution of images
out_dir = "/home/spiderkiller/kitti_dataset/label_2_yolo_format/"

LABEL_MAP = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2}

# Clean output directory 
print("Clean output directory : " + str(out_dir))
rmtree(out_dir, ignore_errors=True)
os.mkdir(out_dir)


file_names = os.listdir(kitti_dir)
file_names.sort()
for file_name in file_names:
    print("Converting " + file_name)
    with open(osp.join(kitti_dir, file_name), 'r') as f:
        h ,w, _= cv2.imread(osp.join(image_dir, file_name.split('.')[0] + '.png')).shape
        s = ""
        dets = f.read().split('\n')
        for det in dets:
            d = det.split()
            try:
                class_name = d[0]
                xmin = float(d[4])
                ymin = float(d[5])
                xmax = float(d[6])
                ymax = float(d[7])
            except IndexError:
                pass
            else:
                if class_name in LABEL_MAP:
                    center_x = (xmin + xmax)/2.0
                    center_y = (ymin + ymax)/2.0
                    bb_w = (xmax - xmin)
                    bb_h = (ymax - ymin)
                    s += class_name + " " + str(center_x/w) + " " + str(center_y/h) + " " + str(bb_w/w) + " " + str(bb_h/h) + '\n'
        
        with open(osp.join(out_dir, file_name), 'w') as out_f:
            out_f.write(s)