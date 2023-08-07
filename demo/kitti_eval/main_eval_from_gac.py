from kitti_common import get_label_annos
from eval import get_official_eval_result
from numba import cuda

# Spiderkiller change file_name to str in order to make it compatible with nuscene_kitti
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    # return [int(line) for line in lines]
    return lines

def eval_from_gac(label_path="/home/hins/Desktop/M3D-RPN/data/kitti/training/label_2",
             result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
             label_split_file="val.txt",
             current_classes=[0],
             gpu='cuda:0',
             dataset_type='kitti'):
    
    cuda.select_device(int(gpu.split(':')[-1]))
    val_image_ids = _read_imageset_file(label_split_file)
    # dt_annos = get_label_annos(result_path)
    dt_annos = get_label_annos(result_path, val_image_ids)
    gt_annos = get_label_annos(label_path, val_image_ids)
    
    # print(f"dt_annos = {dt_annos[0]}")
    # print(f"gt_annos = {gt_annos[0]}")

    result_texts = []
    for current_class in current_classes:
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class, dataset_type))
    return result_texts

if __name__ == "__main__":
    current_classes = [0]
    result_txt = eval_from_gac(label_path="/home/lab530/KenYu/kitti/training/label_2",
                          result_path="/home/lab530/KenYu/pytorch-retinanet/checkpoint/2D_detection_3/result",
                          label_split_file="/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt",
                          current_classes=current_classes,
                          gpu=0,
                          dataset_type='kitti')
    print(result_txt[0])
