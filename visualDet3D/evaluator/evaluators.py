import os
from tqdm import tqdm
from easydict import EasyDict
from typing import Sized

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from visualDet3D.utils.cal import BBox3dProjector, BackProjection
from visualDet3D.utils.kitti_data_parser import write_result_to_file
from visualDet3D.evaluator.kitti_common import get_label_annos
from visualDet3D.evaluator.eval import get_official_eval_result
from numba import cuda

# Spiderkiller change file_name to str in order to make it compatible with nuscene_kitti
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

@torch.no_grad()
def evaluate_kitti_obj(cfg:EasyDict, 
                       model:nn.Module,
                       dataset_val:Sized,
                       writer:SummaryWriter,
                       epoch_num:int,
                       result_path_split='validation',
                       output_path="",
                       ):
    model.eval()

    projector     = BBox3dProjector().cuda()
    backprojector = BackProjection().cuda()
    
    # Added by spiderkiller, able to output <file_name>.txt 
    with open(cfg.data.val_split_file, 'r') as f:
        fn_list = [i for i in f.read().splitlines()]
    assert len(fn_list) == len(dataset_val), f'Number of validation data are not matched. fn_list has {len(fn_list)} files, but dataset_val has {len(dataset_val)} files'
    for index in tqdm(range(len(dataset_val))):
        test_one(cfg, index, fn_list, dataset_val, model, backprojector, projector, output_path)
    
    cuda.select_device(cfg.trainer.gpu)
    val_image_ids = _read_imageset_file(cfg.data.val_split_file)
    dt_annos = get_label_annos(output_path                                      , val_image_ids)
    gt_annos = get_label_annos(os.path.join(cfg.data.train_data_path, 'label_2'), val_image_ids)
    # print(f"dt_annos = {dt_annos[0]}")
    # print(f"gt_annos = {gt_annos[0]}")

    # Evaluation
    result_texts = []
    for current_class in cfg.obj_types:
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class, "kitti", is_ap_crit = False))

    eval_lines = result_texts[0].splitlines()
    bbox_result   = [float(i) for i in eval_lines[1].split(':')[1].split(',')]
    bev_result    = [float(i) for i in eval_lines[2].split(':')[1].split(',')]
    threeD_result = [float(i) for i in eval_lines[3].split(':')[1].split(',')]

    for class_index, result_text in enumerate(result_texts):
        if writer is not None:
            # Add by spderkiller, visulize validation set performance
            writer.add_scalar('val/bev_easy', bev_result[0], epoch_num)
            writer.add_scalar('val/bev_mid',  bev_result[1], epoch_num)
            writer.add_scalar('val/bev_hard', bev_result[2], epoch_num)
            writer.add_scalar('val/3d_easy',  threeD_result[0], epoch_num)
            writer.add_scalar('val/3d_mid',   threeD_result[1], epoch_num)
            writer.add_scalar('val/3d_hard',  threeD_result[2], epoch_num)
            writer.add_text("validation result {}".format(class_index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
        print(result_text)
    
    return (threeD_result[0], threeD_result[1], threeD_result[2],
            bev_result   [0], bev_result   [1], bev_result   [2],
            bbox_result  [0], bbox_result  [1], bbox_result  [2])

def test_one(cfg, index, fn_list, dataset, model, backprojector:BackProjection, projector:BBox3dProjector, result_path):
    
    data = dataset[index]

    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']

    original_height = data['original_shape'][0]
    collated_data = dataset.collate_fn([data])
    height = collated_data[0].shape[2]

    scores, bbox, obj_index = model([collated_data[0].cuda().float().contiguous(), torch.tensor(collated_data[1]).cuda().float()])
    obj_names = [cfg.obj_types[i.item()] for i in obj_index]

    bbox_2d = bbox[:, 0:4]
    if bbox.shape[1] > 4: # run 3D
        bbox_3d_state = bbox[:, 4:] # [cx,cy,z,w,h,l,alpha, bot, top]
        bbox_3d_state_3d = backprojector(bbox_3d_state, P2) #[x, y, z, w,h ,l, alpha, bot, top]

        _, _, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))

        # Recover bbox coordinate via crop and resize
        original_P = data['original_P']
        scale_x = original_P[0, 0] / P2[0, 0]
        scale_y = original_P[1, 1] / P2[1, 1]
        
        shift_left = original_P[0, 2] / scale_x - P2[0, 2]
        shift_top  = original_P[1, 2] / scale_y - P2[1, 2]
        bbox_2d[:, 0:4:2] += shift_left
        bbox_2d[:, 1:4:2] += shift_top

        bbox_2d[:, 0:4:2] *= scale_x
        bbox_2d[:, 1:4:2] *= scale_y

        write_result_to_file(result_path, fn_list[index], scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)
    else:
        if "crop_top" in cfg.data.augmentation and cfg.data.augmentation.crop_top is not None:
            crop_top = cfg.data.augmentation.crop_top
        elif "crop_top_height" in cfg.data.augmentation and cfg.data.augmentation.crop_top_height is not None:
            if cfg.data.augmentation.crop_top_height >= original_height:
                crop_top = 0
            else:
                crop_top = original_height - cfg.data.augmentation.crop_top_height

        scale_2d = (original_height - crop_top) / height
        bbox_2d[:, 0:4] *= scale_2d
        bbox_2d[:, 1:4:2] += cfg.data.augmentation.crop_top
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        write_result_to_file(result_path, fn_list[index], scores, bbox_2d, obj_types = obj_names)
