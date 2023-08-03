import os 

train_fns = []
val_fns   = []

# Get filenames
train_fns = [i.split('.')[0] for i in os.listdir("/home/lab530/KenYu/nusc_kitti/train/image_2/")] 
val_fns   = [i.split('.')[0] for i in os.listdir("/home/lab530/KenYu/nusc_kitti/val/image_2/")] 

# Output train.txt
with open("/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/nuscene_kitti/train.txt", 'w') as f:
    s = ""
    for i in train_fns:
        s += i+'\n'
    f.write(s)
    
# Output val.txt
with open("/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/nuscene_kitti/val.txt", 'w') as f:
    s = ""
    for i in val_fns:
        s += i+'\n'
    f.write(s)
