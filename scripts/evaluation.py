import os
import sys
visualDet3D_path = os.path.dirname(sys.path[0])  #two folders upwards
sys.path.insert(0, visualDet3D_path)

from visualDet3D.evaluator.eval import get_official_eval_result
from visualDet3D.evaluator.kitti_common import get_label_annos
from fire import Fire

# For filter warning
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# # Pseudo-LiDAR
# PRED_PATH = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/pseudo_lidar_prediction"
# # MonoFlex
# PRED_PATH = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/monoflex_prediction/"
# # DD3D
# PRED_PATH = "/home/lab530/KenYu/dd3d/outputs/2cyqwjvr-20220811_163826/inference/final-tta/kitti_3d_val/bbox3d_predictions_standard_format/"

# # GAC original
# PRED_PATH = "../prediction_result/ground_aware_prediction/"
# # SMOKE
# PRED_PATH = "/home/lab530/KenYu/SMOKE/tools/logs/inference/kitti_train/data" 
# "/home/lab530/KenYu/SMOKE/tools/logs/inference/kitti_train/data/"

# PRED_PATH = "/home/lab530/KenYu/MonoGRNet/outputs/kittiBox/val_out/val_result"
# PRED_PATH = "/home/lab530/KenYu/mmdetection/yolov3_exps/output"

# PRED_PATH = "/home/lab530/KenYu/visualDet3D/exp_output/attention/Mono3D/output/validation/data"

# Spiderkiller change file_name to str in order to make it compatible with nuscene_kitti
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

# Add by spiderkiller to allow utilize evaluation function for other directory
def main(label_path, result_path, label_split_file, is_ap_crit):

    current_classes = [0]

    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = get_label_annos(result_path, val_image_ids)
    gt_annos = get_label_annos(label_path, val_image_ids)

    result_texts = []
    for current_class in current_classes:
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class, "kitti", is_ap_crit = is_ap_crit))
    print(result_texts[0])

if __name__ == '__main__':
    Fire(main)