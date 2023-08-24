# # Training
# python scripts/train.py --cfg_path="config/pac/baseline.py"
# python scripts/train.py --cfg_path="config/pac/pac_module.py"
# python scripts/train.py --cfg_path="config/scene-aware/kitti_seg_solid_10_obj_3_zJitter_sceneAware.py"
# python scripts/train.py --cfg_path="config/das/das.py"

# # Evaluate on validation set 
# python ./scripts/test.py --cfg_path="config/pac/pac_module.py" \
#                          --gpu=0 \
#                          --checkpoint_path="exp_output/pac/pac_module/checkpoint/Yolo3D_24.pth" \
#                          --split_to_test="val" \

# # Inference on test set (no label)
# python ./scripts/test.py --cfg_path="config/pac/pac_module.py" \
#                          --gpu=0 \
#                          --checkpoint_path="exp_output/pac/pac_module/checkpoint/Yolo3D_latest.pth" \
#                          --split_to_test="test" \

# # Inference on test sequence (no label)
# python ./scripts/test.py --cfg_path="config/pac/pac_module.py" \
#                          --gpu=0 \
#                          --checkpoint_path="exp_output/pac/pac_module/checkpoint/Yolo3D_latest.pth" \
#                          --split_to_test="test_sequence" \


# # Evaluate prediction file results
# python scripts/evaluation.py \
# --label_path="dataset/kitti/training/label_2" \
# --result_path="prediction_result/ground_aware_prediction/" \
# --label_split_file="visualDet3D/data/kitti_data_split/kitti_anchor_gen_split/val_all.txt" \
# --is_ap_crit=False

# #####################
# ## Sequence2Video ###
# #####################
# python scripts/sequance2video.py \
# --squence_dir="viz_result/kitti_test_sequence" \
# --output_path="viz_result/video/1_5_18.avi"
