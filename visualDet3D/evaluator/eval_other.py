from kitti.evaluate import evaluate

# Pseudo-LiDAR
# PRED_PATH = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/pseudo_lidar_prediction"
# # MonoFlex
# PRED_PATH = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/monoflex_prediction/"
# # DD3D
# PRED_PATH = "/home/lab530/KenYu/dd3d/outputs/2cyqwjvr-20220811_163826/inference/final-tta/kitti_3d_val/bbox3d_predictions_standard_format/"

# This is for producing vliadiont split file 
# ls /home/lab530/KenYu/dd3d/outputs/2cyqwjvr-20220811_163826/inference/final-tta/kitti_3d_val/bbox3d_predictions_standard_format/ -l | awk '{gsub(/\.[^.]*$/,"",$NF); print $NF}'

# # GAC original
PRED_PATH = "/home/lab530/KenYu/visualDet3D/exp_output/baseline_gac_original/Mono3D/output/validation/data/"
# # SMOKE
# PRED_PATH = "/home/lab530/KenYu/SMOKE/tools/logs/inference/kitti_train/data" 
# "/home/lab530/KenYu/SMOKE/tools/logs/inference/kitti_train/data/"

PRED_PATH = "/home/lab530/KenYu/MonoGRNet/outputs/kittiBox/val_out/val_result"
# PRED_PATH = "/home/lab530/KenYu/mmdetection/yolov3_exps/output"

# PRED_PATH = "/home/lab530/KenYu/visualDet3D/exp_output/attention/Mono3D/output/validation/data"
# Add by spiderkiller to allow utilize evaluation function for other directory
if __name__ == "__main__":
    current_classes = [0]
    result_txt = evaluate(label_path="/home/lab530/KenYu/kitti/training/label_2",
                          result_path=PRED_PATH,
                        #   label_split_file = "/home/lab530/KenYu/visualDet3D/visualDet3D/evaluator/dd3d_val_split.txt",
                          label_split_file="/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt",
                          current_classes=current_classes,
                          gpu=0,
                          dataset_type='kitti',
                          is_ap_crit = False)
    print(result_txt[0])
