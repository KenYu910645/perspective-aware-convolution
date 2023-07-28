''' KITTI dataset
<dataset>
    |
    |---- train
            |
            |---- image
            |---- label
            |---- train.txt
    |---- val
            |
            |---- image
            |---- label
            |---- val.txt
    |---- test
            |
            |---- image
            |---- label
            |---- test.txt
'''
import os.path as osp
import os
import cv2
import random
from collections import namedtuple
from shutil import copyfile, rmtree

############
### TODO ###
############

# Input
img_dir = "/home/spiderkiller/kitti_dataset/image_2/"
ano_dir = "/home/spiderkiller/kitti_dataset/label_2/"
# Output
dataset_dir = "/home/spiderkiller/kitti_dataset/"
# Parameters
data_pp = {'train' : 70, 'val' : 15, 'test' : 15} # Data_propotion
assert data_pp['train'] + data_pp['val'] + data_pp['test'] == 100, "propotion of train + val + test need to be 100"


#############################
###  Split train val test ###
#############################
# remove everything that's in output directory
rmtree(dataset_dir + "train/", ignore_errors=True)
rmtree(dataset_dir + "val/",   ignore_errors=True)
rmtree(dataset_dir + "test/",  ignore_errors=True)
print("Deleted directories train/, val/ and test/")

# Create directory stucture
for i in ['train/', 'train/image/', 'train/label/',
          'val/',   'val/image/',   'val/label/',
          'test/',  'test/image/', 'test/label/']:
    os.mkdir(dataset_dir + i)
    print("Create directory : " + dataset_dir + i)

# Load inputs
img_paths = [osp.join(img_dir, i) for i in os.listdir(img_dir)]
ano_paths = [osp.join(ano_dir, i) for i in os.listdir(ano_dir)]
print("Get " + str(len(img_paths)) + " images in " + img_dir)
print("Get " + str(len(ano_paths)) + " labels in " + ano_dir)
num = len(img_paths)

# Check images and annotations are well corresspond.
for img_path in img_paths:
    n = osp.split(img_path)[1].split('.')[0]
    if not osp.isfile(osp.join(ano_dir, n + '.txt')):
        print("Can't find file " + str(osp.join(ano_dir, n + '.txt')))
        print("split abort.")
print("Successfully checked input images and labels")

# Shuffle images order
random.shuffle(img_paths)

# Split 
train_imgs = []
val_imgs   = []
test_imgs  = []
for img_path in img_paths:
    if len(train_imgs) <= int(num*data_pp['train']/100):
        train_imgs.append(img_path)
    elif len(val_imgs) <= int(num*data_pp['val']/100):
        val_imgs.append(img_path)
    else:
        test_imgs.append(img_path)

train_imgs.sort()
val_imgs.sort()
test_imgs.sort()
print("Split results(train/val/test): " + str(len(train_imgs)) + "/" + str(len(val_imgs)) + "/" + str(len(test_imgs)))

# Copy images
print("Coping " + str(len(train_imgs)) + " images to " + dataset_dir + "train/image/")
[copyfile(i, osp.join(dataset_dir + "train/image/", osp.split(i)[1])) for i in train_imgs]
print("Coping " + str(len(val_imgs))   + " images to " + dataset_dir + "val/image/")
[copyfile(i, osp.join(dataset_dir + "val/image/",   osp.split(i)[1])) for i in val_imgs]
print("Coping " + str(len(test_imgs))  + " images to " + dataset_dir + "test/image/")
[copyfile(i, osp.join(dataset_dir + "test/image/",  osp.split(i)[1])) for i in test_imgs]

# Copy txt
print("Coping " + str(len(train_imgs)) + " labels to " + dataset_dir + "train/label/")
[copyfile(osp.join(ano_dir, osp.split(i)[1].split('.')[0] + '.txt'), osp.join(dataset_dir + "train/label/", osp.split(i)[1].split('.')[0] + '.txt')) for i in train_imgs]
print("Coping " + str(len(val_imgs))   + " labels to " + dataset_dir + "val/label/")
[copyfile(osp.join(ano_dir, osp.split(i)[1].split('.')[0] + '.txt'), osp.join(dataset_dir + "val/label/", osp.split(i)[1].split('.')[0] + '.txt')) for i in val_imgs]
print("Coping " + str(len(test_imgs))  + " labels to " + dataset_dir + "test/label/")
[copyfile(osp.join(ano_dir, osp.split(i)[1].split('.')[0] + '.txt'), osp.join(dataset_dir + "test/label/", osp.split(i)[1].split('.')[0] + '.txt')) for i in test_imgs]

# Create index txt
s = [osp.split(i)[1].split('.')[0] +'\n' for i in train_imgs]
with open(osp.join(dataset_dir + 'train/', 'train.txt'), 'w') as f: f.write(''.join(s))
s = [osp.split(i)[1].split('.')[0] +'\n' for i in val_imgs]
with open(osp.join(dataset_dir + 'val/',   'val.txt'), 'w') as f: f.write(''.join(s))
s = [osp.split(i)[1].split('.')[0] +'\n' for i in test_imgs]
with open(osp.join(dataset_dir + 'test/',  'test.txt'), 'w') as f: f.write(''.join(s))