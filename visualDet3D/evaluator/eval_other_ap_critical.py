
from kitti.evaluate import evaluate

# For filter warning
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Add by spiderkiller to allow utilize evaluation function for other directory
if __name__ == "__main__":
    current_classes = [0]
    
    # Original AP
    result_txt = evaluate(label_path="/home/lab530/KenYu/kitti/training/label_2",
                          result_path="/home/lab530/KenYu/visualDet3D/exp_output/best/Mono3D/output/validation/data",
                          label_split_file="/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt",
                          current_classes=current_classes,
                          gpu=0,
                          dataset_type='kitti',
                          is_ap_crit = False,)
    print(result_txt[0])
    
    # AP_crit
    result_txt = evaluate(label_path="/home/lab530/KenYu/kitti/training/label_2",
                          result_path="/home/lab530/KenYu/visualDet3D/exp_output/best/Mono3D/output/validation/data",
                          label_split_file="/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt",
                          current_classes=current_classes,
                          gpu=0,
                          dataset_type='kitti',
                          is_ap_crit = True,)
    print(result_txt[0])