import io as sysio

import numba
import numpy as np
from scipy.interpolate import interp1d

from .rotate_iou import rotate_iou_gpu_eval
from visualDet3D.ap_critical.ap_critical import cal_criticality

@numba.jit(nopython=True)
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty, dataset_type):
    '''
    Cleaning data according to their difficulty
    '''
    if dataset_type == "nuscene":
        CLASS_NAMES = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 
                       'pedestrian', 'traffic_cone', 'trailer', 'truck']
    else:
        CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'car',
                       'tractor', 'trailer']
    MIN_HEIGHT     = [40, 25, 25, 25]
    MAX_OCCLUSION  = [0, 1, 2, 99]
    MAX_TRUNCATION = [0.15, 0.3, 0.5, 99]
    
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes,
                          qboxes,
                          rinc,
                          criterion=-1,
                          z_axis=1,
                          z_center=1.0):
    """
        z_axis: the z (height) axis.
        z_center: unified z (height) center of box.
    """
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                min_z = min(
                    boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                    qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center))
                max_z = max(
                    boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                    qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center)
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """kitti camera format z_axis=1.
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    rinc = rotate_iou_gpu_eval(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False,
                           is_ap_crit=False):

    NO_DETECTION = -10000000
    
    dt_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    #
    dt_bboxes = dt_datas[:, :4]
    dt_alphas = dt_datas[:,  4]
    dt_scores = dt_datas[:,  5]
    dt_kappas = dt_datas[:,  6]
    gt_alphas = gt_datas[:,  4]
    gt_kappas = gt_datas[:,  5]
    
    # print(dt_kappas)
    # print(gt_kappas)

    # print(f"(dt_size, gt_size) = {(dt_size, gt_size)}")

    # print(f"ignored_gt = {ignored_gt}")   # 0 means don't ignore it, or 1 or 
    #                                        -1 means totally ignore it 
    #                                        
    # print(f"ignored_det = {ignored_det}") # 0 or 1
    
    assigned_detection = [False] * dt_size
    ignored_threshold  = [False] * dt_size
    if compute_fp:
        for i in range(dt_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    tp, fp, fn, sim = 0, 0, 0, 0
    tp_crit, fp_crit, fn_crit, sim_crit = 0, 0, 0, 0 # For AP_crit 
    # TODO here should be tp_gt_crit and tp_dt_crit
    
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta       = np.zeros((gt_size, ))
    delta_kappa = np.zeros((gt_size, ))
    delta_idx = 0
    for i_gt in range(gt_size):
        if ignored_gt[i_gt] == -1: continue # Completely ignore this groundtrue
        
        det_idx = -1
        valid_detection = NO_DETECTION # Try to find assignment for this gt
        max_overlap = 0
        assigned_ignored_det = False
        
        # Traverse all detection and find the best assignment
        for i_dt in range(dt_size):
            
            # Skip the detection if it's ignore or it has assigned to other gt
            if (ignored_det[i_dt] == -1) or (assigned_detection[i_dt]) or (ignored_threshold[i_dt]): continue
            
            overlap  = overlaps[i_dt, i_gt]
            dt_score = dt_scores[i_dt]
            
            if overlap > min_overlap:
                det_idx = i_dt
                if compute_fp:
                    if (overlap > max_overlap or assigned_ignored_det) and ignored_det[i_dt] == 0:
                        max_overlap = overlap
                        valid_detection = 1
                        assigned_ignored_det = False
                    
                    if (valid_detection == NO_DETECTION) and ignored_det[i_dt] == 1:
                        valid_detection = 1
                        assigned_ignored_det = True
                else:
                    if dt_score > valid_detection:
                        valid_detection = dt_score

        if valid_detection == NO_DETECTION:
            if ignored_gt[i_gt] == 0: # False Negative
                fn      += 1
                fn_crit += gt_kappas[i_gt]
            
        else:
            if ignored_gt[i_gt] == 1 or ignored_det[det_idx] == 1:
                assigned_detection[det_idx] = True
            else:
                # Only when ignored_gt[i_gt] == 0 and ignored_det[det_idx] == 0, than, add a tp and a threshold.
                tp      += 1
                tp_crit += gt_kappas[i_gt]
                thresholds[thresh_idx] = dt_scores[det_idx]
                thresh_idx += 1
                if compute_aos:
                    delta[delta_idx] = gt_alphas[i_gt] - dt_alphas[det_idx]
                    delta_kappa[delta_idx] = dt_kappas[det_idx] # For AP_crit
                    delta_idx += 1

                assigned_detection[det_idx] = True
    
    if compute_fp:
        for i_dt in range(dt_size):
            if (not (assigned_detection[i_dt] or ignored_det[i_dt] == -1
                     or ignored_det[i_dt] == 1 or ignored_threshold[i_dt])):
                fp      += 1
                fp_crit += dt_kappas[i_dt]
        
        # If the detection box overlap with dontcare bbox, than don't count it as false positive
        nstuff = 0
        nstuff_crit = 0
        if metric == 0: # for 2d bounding box
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(dt_size):
                    if (assigned_detection[j]) or (ignored_det[j] == -1 or ignored_det[j] == 1) or (ignored_threshold[j]): continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff      += 1
                        nstuff_crit += dt_kappas[j]
        fp      -= nstuff
        fp_crit -= nstuff_crit
        
        if compute_aos:
            tmp      = np.zeros((fp + delta_idx, ))
            tmp_crit = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp     [i + fp] = ( (1.0 + np.cos(delta[i])) / 2.0 )
                tmp_crit[i + fp] = ( (1.0 + np.cos(delta[i])) / 2.0 ) * delta_kappa[i]
                
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                sim      = np.sum(tmp)
                sim_crit = np.sum(tmp_crit)
            else:
                sim = -1
                sim_crit = -1
    # print(f"(tp, fp, fn) = {((tp, fp, fn))}")
    # print(f"(tp_crit, fp_crit, fn_crit) = {((tp_crit, fp_crit, fn_crit))}")
    if is_ap_crit:
        return tp_crit, fp_crit, fn_crit, sim_crit, thresholds[:thresh_idx]
    else:
        return tp, fp, fn, sim, thresholds[:thresh_idx]

def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False,
                             is_ap_crit=False):
    # pr is the output of this function
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num +
                               gt_nums[i]]

            gt_data     = gt_datas    [gt_num:gt_num + gt_nums[i]]
            dt_data     = dt_datas    [dt_num:dt_num + dt_nums[i]]
            ignored_gt  = ignored_gts [gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare    = dontcares   [dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos,
                is_ap_crit=is_ap_crit)
            # print(f"tp, fp, fn, similarity = {(tp, fp, fn, similarity)}")
            
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1: # For AOS
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos,
                         dt_annos,
                         metric,
                         num_parts=50,
                         z_axis=1,
                         z_center=1.0):
    """fast iou algorithm. this function can be used independently to
    do result analysis. 
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
        z_axis: height axis. kitti camera use 1, lidar use 2.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    bev_axes = list(range(3))
    bev_axes.pop(z_axis)
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        
        # Find 2D bounding box overlap
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        
        # Find BEV bounding box overlap
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = bev_box_overlap(gt_boxes,
                                           dt_boxes).astype(np.float64)

        # Find 3D bounding box overlap
        elif metric == 2:
            loc  = np.concatenate([a["location"]   for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            loc  = np.concatenate([a["location"]   for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            overlap_part = d3_box_overlap(
                gt_boxes, dt_boxes, z_axis=z_axis,
                z_center=z_center).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx +
                                   gt_box_num, dt_num_idx:dt_num_idx +
                                   dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty, dataset_type):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dts, dontcares = [], [], []
    total_num_valid_gt = 0
    
    for i_img in range(len(gt_annos)):
        num_valid_gt, ignored_gt, ignored_dt, dc_bboxes = clean_data(gt_annos[i_img], dt_annos[i_img], current_class, difficulty, dataset_type)
        
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dts.append(np.array(ignored_dt, dtype=np.int64))
        
        # Don't care ground true
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        
        total_num_valid_gt += num_valid_gt
        
        ###############
        ### AP_crit ###
        ############### Add by spiderkiller
        xzyaw_gt = np.concatenate([ gt_annos[i_img]["location"][:, :1], 
                                    gt_annos[i_img]["location"][:, 2:3],  
                                    gt_annos[i_img]["rotation_y"][..., np.newaxis]], 1)
        gt_kappa = cal_criticality(xzyaw_gt, ap_mode="my")
        xzyaw_dt = np.concatenate([ dt_annos[i_img]["location"][:, :1], 
                                    dt_annos[i_img]["location"][:, 2:3],  
                                    dt_annos[i_img]["rotation_y"][..., np.newaxis]], 1)
        dt_kappa = cal_criticality(xzyaw_dt, ap_mode="my")
        # print(gt_annos[i_img]["alpha"].shape)
        # print(dt_kappa)
        
        # gt_datas = np.concatenate([gt_annos[i_img]["bbox"], gt_annos[i_img]["alpha"][..., np.newaxis]], 1)
        # dt_datas = np.concatenate([dt_annos[i_img]["bbox"], dt_annos[i_img]["alpha"][..., np.newaxis], dt_annos[i_img]["score"][..., np.newaxis]], 1)
        gt_datas = np.concatenate([gt_annos[i_img]["bbox"], gt_annos[i_img]["alpha"][..., np.newaxis], gt_kappa[..., np.newaxis]], 1)
        dt_datas = np.concatenate([dt_annos[i_img]["bbox"], dt_annos[i_img]["alpha"][..., np.newaxis], dt_annos[i_img]["score"][..., np.newaxis], dt_kappa[..., np.newaxis]], 1)
        
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    
    total_dc_num = np.stack(total_dc_num, axis=0)
    
    return (gt_datas_list, dt_datas_list, 
            ignored_gts, ignored_dts, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
                  dt_annos,
                  current_classes,
                  difficultys,
                  metric,
                  min_overlaps,
                  compute_aos=False,
                  z_axis=1,
                  z_center=1.0,
                  num_parts=50,
                  dataset_type='kitti',
                  is_ap_crit=False):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: list, must from get_label_annos() in kitti_common.py, [3769]
            gt_annos[0] = {'name': ['Car', ... ], 
                           'truncated': [0, -1, ....]
                           'occluded': [0, -1, ....]
                           'alpha' : [-1.57, ...]
                           'bbox' : [n_gt, 4]
                           'dimensions': [n_gt, 3]
                           'location': [n_gt, 3],
                           'rotation_y':[-1.56, ....],
                           'score': [0, ...]}
        
        dt_annos: list, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official: 
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]] 
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    # print(f"[eval.py] gt_annos = {gt_annos.__len__()}") # 3769
    # print(f"[eval.py] dt_annos = {dt_annos.__len__()}") # 3769
    # print(f"[eval.py] current_classes = {current_classes}") # [0]
    # print(f"[eval.py] difficultys = {difficultys}") # [0, 1, 2]
    # print(f"[eval.py] metric = {metric}") # 0
    # print(f"[eval.py] min_overlaps = {min_overlaps}")
    # print(gt_annos[0])
    
    assert len(gt_annos) == len(dt_annos)
    # print(f"[eval.py] compute_aos = {compute_aos}")
    if is_ap_crit:
        assert compute_aos, "[eval.py] currently don't support compute_aos = False"
    N_SAMPLE_PTS = 41
    
    split_parts = get_split_parts(len(gt_annos), num_parts) # [75, 75, 75, 75, ... 19], split frame into 75, 75, ... parts
    # print(f"split_parts = {np.sum(split_parts)}") # 3769
    
    (overlaps, # list [3769] (n_detection, n_groundtrue)
     parted_overlaps, # list [51] (282, 517)
     total_dt_num, # np.array, (3769,) # How many detections are there in this frame
     total_gt_num, # np.array, (3769,) # How many groundtrue are there in this frame
     ) = calculate_iou_partly(dt_annos,
                              gt_annos,
                              metric,
                              num_parts,
                              z_axis=z_axis,
                              z_center=z_center)
    
    # Initalize everything
    n_lap = len(min_overlaps)
    n_cls = len(current_classes)
    n_dif = len(difficultys)
    precision      = np.zeros([n_cls, n_dif, n_lap, N_SAMPLE_PTS])
    recall         = np.zeros([n_cls, n_dif, n_lap, N_SAMPLE_PTS])
    aos            = np.zeros([n_cls, n_dif, n_lap, N_SAMPLE_PTS])
    all_thresholds = np.zeros([n_cls, n_dif, n_lap, N_SAMPLE_PTS])
    
    for i_cls, current_class in enumerate(current_classes):
        for i_dif, difficulty in enumerate(difficultys):
            
            (gt_datas_list, # list [3769][n_gt ,5] (x1, y1, x2, y2, alpha)
             dt_datas_list, # list [3769][n_dt ,6] (x1, y1, x2, y2, alpha, score)
             ignored_gts,   # [3769], ] [-1  1 -1 -1 -1 -1 -1]
             ignored_dets,  # [3769], ] [1]
             dontcares,     # [3769][n_dontcare, 4] (x1, y1, x2, y2)
             total_dc_num,  # [3769], ]
             total_num_valid_gt, # int
             ) = _prepare_data(gt_annos, dt_annos, current_class, difficulty, dataset_type)
            
            # print(f"gt_datas_list = {(gt_datas_list[0])}")
            # print(f"dt_datas_list = {dt_datas_list[0]}")
            # print(f"ignored_gts = {(ignored_gts[0])}")
            # print(f"ignored_dets = {(ignored_dets[0])}")
            # print(f"dontcares = {(dontcares)[0]}")
            # print(f"total_dc_num = {(total_dc_num)}")
            # print(f"total_num_valid_gt = {(total_num_valid_gt)}") # int
            
            for i_lap, min_overlap in enumerate(min_overlaps[:, metric, i_cls]):
                
                thresholdss = []
                for i_img in range(len(gt_annos)):
                    
                    # Get thresholds
                    _, _, _, _, thresholds = compute_statistics_jit(overlaps[i_img],
                                                                    gt_datas_list[i_img],
                                                                    dt_datas_list[i_img],
                                                                    ignored_gts[i_img],
                                                                    ignored_dets[i_img],
                                                                    dontcares[i_img],
                                                                    metric,
                                                                    min_overlap=min_overlap,
                                                                    thresh=0.0,
                                                                    compute_fp=False,
                                                                    is_ap_crit=is_ap_crit)
                    thresholdss += thresholds.tolist()
                
                thresholds = np.array( get_thresholds(np.array(thresholdss), total_num_valid_gt) )
                
                # print(f"thresholds = {thresholds}")
                all_thresholds[i_cls, i_dif, i_lap, :len(thresholds)] = thresholds
                
                pr = np.zeros([len(thresholds), 4]) # (tp, fp, fn , aos)
                idx = 0
                for i_part, n_img in enumerate(split_parts): # only deal with 75 image at a time
                    gt_datas_part     = np.concatenate(gt_datas_list[idx:idx + n_img], 0)
                    dt_datas_part     = np.concatenate(dt_datas_list[idx:idx + n_img], 0)
                    dc_datas_part     = np.concatenate(dontcares    [idx:idx + n_img], 0)
                    ignored_dets_part = np.concatenate(ignored_dets [idx:idx + n_img], 0)
                    ignored_gts_part  = np.concatenate(ignored_gts  [idx:idx + n_img], 0)
                    
                    fused_compute_statistics(
                        parted_overlaps[i_part],
                        pr,
                        total_gt_num[idx:idx + n_img],
                        total_dt_num[idx:idx + n_img],
                        total_dc_num[idx:idx + n_img],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=True,
                        is_ap_crit=is_ap_crit)
                    idx += n_img

                for i_thres in range(len(thresholds)):
                    precision[i_cls, i_dif, i_lap, i_thres] = pr[i_thres, 0] / (pr[i_thres, 0] + pr[i_thres, 1]) # precision = tp / (tp+fp)
                    aos      [i_cls, i_dif, i_lap, i_thres] = pr[i_thres, 3] / (pr[i_thres, 0] + pr[i_thres, 1])
                
                for i_thres in range(len(thresholds)):
                    precision[i_cls, i_dif, i_lap, i_thres] = np.max(precision[i_cls, i_dif, i_lap, i_thres:], axis=-1)
                    aos      [i_cls, i_dif, i_lap, i_thres] = np.max(aos      [i_cls, i_dif, i_lap, i_thres:], axis=-1)

    ret_dict = {
        # "recall": recall, # [num_cls, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
        "precision": precision, # (1, 3, 2, 41)
        "orientation": aos, # (1, 3, 2, 41)
        "thresholds": all_thresholds, # (1, 3, 2, 41)
        "min_overlaps": min_overlaps, # (2, 3, 1)
    }  
    
    return ret_dict


def get_mAP_v2(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def do_eval_v2(gt_annos,
               dt_annos,
               current_classes,
               min_overlaps,
               compute_aos=False,
               difficultys=(0, 1, 2),
               z_axis=1,
               z_center=1.0):
    # min_overlaps: [num_minoverlap, metric, num_class]
    ret = eval_class(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        0,
        min_overlaps,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP_v2(ret["precision"])
    mAP_aos = None
    if compute_aos:
        mAP_aos = get_mAP_v2(ret["orientation"])
    ret = eval_class(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        1,
        min_overlaps,
        z_axis=z_axis,
        z_center=z_center)
    mAP_bev = get_mAP_v2(ret["precision"])
    ret = eval_class(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        2,
        min_overlaps,
        z_axis=z_axis,
        z_center=z_center)
    mAP_3d = get_mAP_v2(ret["precision"])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos

def do_eval_v3(gt_annos,
               dt_annos,
               current_classes,
               min_overlaps,
               compute_aos=False,
               difficultys=(0, 1, 2),
               z_axis=1,
               z_center=1.0,
               dataset_type='kitti',
               is_ap_crit=False):
    # min_overlaps: [num_minoverlap, metric, num_class]
    types = ["bbox", "bev", "3d"]
    metrics = {}
    for i in range(3):
        ret = eval_class(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            i,
            min_overlaps,
            compute_aos,
            z_axis=z_axis,
            z_center=z_center,
            dataset_type=dataset_type,
            is_ap_crit=is_ap_crit,)
        metrics[types[i]] = ret
    return metrics


def do_coco_style_eval(gt_annos,
                       dt_annos,
                       current_classes,
                       overlap_ranges,
                       compute_aos,
                       z_axis=1,
                       z_center=1.0):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval_v2(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

def get_official_eval_result(gt_annos,
                             dt_annos,
                             current_classes,
                             dataset_type='kitti',
                             difficultys=[0, 1, 2],
                             z_axis=1,
                             z_center=1.0,
                             is_ap_crit=False):
    """
        gt_annos and dt_annos must contains following keys:
        [bbox, location, dimensions, rotation_y, score]
    """
    if dataset_type == "nuscene":
        overlap_mod = np.array([[0.5, 0.5, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.7, 0.7],
                                [0.5, 0.5, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.7, 0.7],
                                [0.5, 0.5, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.7, 0.7]])
        overlap_easy = np.array([[0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7],
                                 [0.25, 0.25, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.5, 0.5],
                                 [0.25, 0.25, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.5, 0.5]])
    else:
        overlap_mod = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7]])
        overlap_easy = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
                                [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
                                [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5]])
    min_overlaps = np.stack([overlap_mod, overlap_easy], axis=0)  # [2, 3, 5]
    print(f"dataset_type in innest function : {dataset_type}")
    if dataset_type == "nuscene":
        class_to_name = {
            0: 'barrier',
            1: 'bicycle',
            2: 'bus',
            3: 'car',
            4: 'construction_vehicle',
            5: 'motorcycle',
            6: 'pedestrian',
            7: 'traffic_cone',
            8: 'trailer',
            9: 'truck',
        }
    else:
        class_to_name = {
            0: 'Car',
            1: 'Pedestrian',
            2: 'Cyclist',
            3: 'Van',
            4: 'Person_sitting',
            5: 'car',
            6: 'tractor',
            7: 'trailer',
        }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    print(f"current_classes = {current_classes}") # [0]
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    metrics = do_eval_v3(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos,
        difficultys,
        z_axis=z_axis,
        z_center=z_center,
        dataset_type=dataset_type,
        is_ap_crit=is_ap_crit)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            mAPbbox = get_mAP_v2(metrics["bbox"]["precision"][j, :, i])
            mAPbbox = ", ".join(f"{v:.2f}" for v in mAPbbox)
            mAPbev = get_mAP_v2(metrics["bev"]["precision"][j, :, i])
            mAPbev = ", ".join(f"{v:.2f}" for v in mAPbev)
            mAP3d = get_mAP_v2(metrics["3d"]["precision"][j, :, i])
            mAP3d = ", ".join(f"{v:.2f}" for v in mAP3d)
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str(f"bbox AP:{mAPbbox}")
            result += print_str(f"bev  AP:{mAPbev}")
            result += print_str(f"3d   AP:{mAP3d}")
            if compute_aos:
                mAPaos = get_mAP_v2(metrics["bbox"]["orientation"][j, :, i])
                mAPaos = ", ".join(f"{v:.2f}" for v in mAPaos)
                result += print_str(f"aos  AP:{mAPaos}")


    return result


def get_coco_eval_result(gt_annos,
                         dt_annos,
                         current_classes,
                         z_axis=1,
                         z_center=1.0):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    class_to_range = {
        0: [0.5, 1.0, 0.05],
        1: [0.25, 0.75, 0.05],
        2: [0.25, 0.75, 0.05],
        3: [0.5, 1.0, 0.05],
        4: [0.25, 0.75, 0.05],
        5: [0.5, 1.0, 0.05],
        6: [0.5, 1.0, 0.05],
        7: [0.5, 1.0, 0.05],
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
        5: [0.5, 0.95, 10],
        6: [0.5, 0.95, 10],
        7: [0.5, 0.95, 10],
    }

    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos,
        dt_annos,
        current_classes,
        overlap_ranges,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result
