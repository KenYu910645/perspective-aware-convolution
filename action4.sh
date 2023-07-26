EXP_NAME=('baseline')

for exp_name in "${EXP_NAME[@]}"
do
    ./launchers/det_precompute.sh config/baseline.py train
    ./launchers/train.sh config/baseline.py 1 "$exp_name"
done

# EXP_NAME=('cbam_only_spatial')

# for exp_name in "${EXP_NAME[@]}"
# do
#     ./launchers/det_precompute.sh config/attention/"$exp_name".py train
#     ./launchers/train.sh config/attention/"$exp_name".py 1 "$exp_name"
# done

# ./launchers/det_precompute.sh config/fpn_3d/fpn_2d_pretrain.py train
# ./launchers/train.sh config/fpn_3d/fpn_2d_pretrain.py 0 fpn_2d_pretrain

# ./launchers/det_precompute.sh config/fpn_3d/baseline_pixelwise_anchor_hack.py train
# ./launchers/train.sh config/fpn_3d/baseline_pixelwise_anchor_hack.py 0 baseline_pixelwise_anchor_hack

# ./launchers/det_precompute.sh config/anchor_gen/gac_original.py train
# ./launchers/train.sh config/anchor_gen/gac_original.py 0 gac_original

# ./launchers/train.sh config/anchor_gen/gac_original.py 0 fpn_3d_gac_original

# NMS
# ./launchers/det_precompute.sh config/nms_test/nms_0.py train
# ./launchers/train.sh config/nms_test/nms_0.py 1 nms_0 > exp_output/nms_test/nms_0/screen_output.txt
# ./launchers/det_precompute.sh config/nms_test/nms_0_25.py train
# ./launchers/train.sh config/nms_test/nms_0_25.py 1 nms_0_25 > exp_output/nms_test/nms_0_25/screen_output.txt
# ./launchers/det_precompute.sh config/nms_test/nms_0_5.py train
# ./launchers/train.sh config/nms_test/nms_0_5.py 1 nms_0_5 > exp_output/nms_test/nms_0_5/screen_output.txt
# ./launchers/det_precompute.sh config/nms_test/nms_0_75.py train
# ./launchers/train.sh config/nms_test/nms_0_75.py 1 nms_0_75 > exp_output/nms_test/nms_0_75/screen_output.txt
# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation
# ./launchers/eval.sh config/nuscene_kitti.py 0 /home/lab530/KenYu/visualDet3D/exp_output/nuscene_kitti/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set( a single image)
# ./launchers/det_precompute.sh config/mixup/kitti_mixup_1.py test
# ./launchers/eval.sh config/mixup/kitti_mixup_1.py 0 /home/lab530/KenYu/visualDet3D/exp_output/mixup/kitti_mixup_1/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py


#########################
### Anchor generation ###
#########################
# ./launchers/det_precompute.sh config/gac_original.py train
# ./launchers/train.sh config/gac_original.py 0 gac_original 

# Train for anchor gen experiment
# ./launchers/det_precompute.sh config/anchor_gen.py train
# ./launchers/train.sh config/anchor_gen.py 0 anchor_gen 

# ./launchers/det_precompute.sh config/gac_original.py train
# ./launchers/train.sh config/gac_original.py 0 gac_original

# Evaluation on validation set
# ./launchers/det_precompute.sh config/anchor_gen.py train # test
# ./launchers/eval.sh config/anchor_gen.py 0 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/BevAnkYolo3D_latest.pth validation
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py

# Test one 
# ./launchers/det_precompute.sh config/anchor_gen.py test
# ./launchers/eval.sh config/anchor_gen.py 0 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/BevAnkYolo3D_latest.pth test
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py


