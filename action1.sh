
# ./launchers/det_precompute.sh config/iou/iou.py train
# ./launchers/train.sh config/iou/iou.py 1 iou

EXP_NAME=('das_test')

for exp_name in "${EXP_NAME[@]}"
do
    ./launchers/det_precompute.sh config/das/"$exp_name".py train
    # ./launchers/train.sh config/das/"$exp_name".py 0 "$exp_name"
done

# Preprocessing
# ./launchers/det_precompute.sh config/data_augumentation/baseline.py train
# Training 
# ./launchers/train.sh config/data_augumentation/baseline.py 1 baseline > exp_output/data_augumentation/baseline/screen_output.txt
# ./launchers/det_precompute.sh config/data_augumentation/add_right_img.py train
# ./launchers/det_precompute.sh config/nuscene_kitti.py train


# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation
# ./launchers/eval.sh config/nuscene_kitti.py 0 /home/lab530/KenYu/visualDet3D/exp_output/nuscene_kitti/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set

# ./launchers/det_precompute.sh config/attention/coordatten.py train
# ./launchers/eval.sh config/attention/coordatten.py 0 

# ./launchers/det_precompute.sh config/gac_original.py train
# ./launchers/train.sh config/gac_original.py 0 gac_original
