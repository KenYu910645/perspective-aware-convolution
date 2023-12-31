import numpy as np
import os
import pickle
import cv2
from copy import deepcopy
import torch

from visualDet3D.networks.detectors.anchors import Anchors, load_from_pkl_or_npy, generate_anchors
from visualDet3D.utils.cal import calc_iou
from visualDet3D.utils.kitti_data_parser import KittiData
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.util_kitti import kitti_calib_file_parser
from visualDet3D.data_augmentation.copy_paste import CopyPaste_Object
from visualDet3D.data_augmentation.augmentation_composer import AugmentataionComposer
from visualDet3D.utils.registry import AUGMENTATION_DICT

def process_train_val_file(cfg):

    with open(cfg.data.train_split_file) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            train_lines[i] = train_lines[i].strip()

    with open(cfg.data.val_split_file) as f:
        val_lines = f.readlines()
        for i  in range(len(val_lines)):
            val_lines[i] = val_lines[i].strip()

    return train_lines, val_lines

def preprocess_val_dataset(cfg, index_names, data_root_dir, output_dict):

    print("start reading validation dataset")

    frames = [None] * len(index_names)
    
    for i, index_name in enumerate(index_names):
        
        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, image, label, velo, depth = data_frame.read_data()

        data_frame.label = [obj for obj in label.data if obj.type in cfg.obj_types]
        
        # Load calibration file
        data_frame.calib = calib

        # Save label.txt and calib.txt in frames[ KittiData(), KittiData(), KittiData(), ...]
        frames[i] = data_frame
    
    print("validation split finished precomputing")
    return frames

def preprocess_train_dataset(cfg, index_names, data_root_dir, output_dict, time_display_inter=100):

    N = len(index_names)
    frames = [None] * N
    print("start reading training data")
    timer = Timer()
    
    external_anchor_path = cfg.detector.anchors.external_anchor_path
    anchor_prior         = cfg.detector.anchors.anchor_prior
    
    total_objects        = [0 for _ in range(len(cfg.obj_types))]
    total_usable_objects = [0 for _ in range(len(cfg.obj_types))]
    
    if external_anchor_path != "":
        anchor_fns = [os.path.join(external_anchor_path, fns) for fns in os.listdir(external_anchor_path)]
        bbox2d = load_from_pkl_or_npy( next(f for f in anchor_fns if "2dbbox" in f) )
        num_external_anchor = bbox2d.shape[0]
        # print(f"bbox2d = {bbox2d.shape}") # (32, 4)
    else:
        bbox2d = generate_anchors(base_size = cfg.detector.anchors.sizes[0], 
                                  ratios    = cfg.detector.anchors.ratios, 
                                  scales    = cfg.detector.anchors.scales)
    
    # Initalize mean_std things
    if anchor_prior:
        anchor_manager = Anchors(cfg, is_training_process=False)
        
        # preprocess = build_augmentator(cfg.data.test_augmentation)
        preprocess = AugmentataionComposer(cfg.data.test_augmentation)

        total_objects        = [0 for _ in range(len(cfg.obj_types))]
        total_usable_objects = [0 for _ in range(len(cfg.obj_types))]
        
        len_scale  = len(anchor_manager.scales)
        len_ratios = len(anchor_manager.ratios)
        len_level  = len(anchor_manager.pyramid_levels) if not cfg.detector.head.is_das else 1
        
        if external_anchor_path == "": 
            num_covered_gt = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios]) # [1, 16, 2]
            sum_covered_gt = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3])  # [z, sin, cos]
            squ_covered_gt = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3], dtype=np.float64)
            # print(f"num_covered_gt = {num_covered_gt.shape}]") # (1, 48, 2)
        else:
            num_covered_gt = np.zeros([len(cfg.obj_types), num_external_anchor]) # [1, 32]
            sum_covered_gt = np.zeros([len(cfg.obj_types), num_external_anchor, 3]) # [1, 32, 3],  [z, sin, cos]
            squ_covered_gt = np.zeros([len(cfg.obj_types), num_external_anchor, 3], dtype=np.float64)
            
        sum_zscwhl = np.zeros((len(cfg.obj_types), 6), dtype=np.float64) # [z, sin2a, cos2a, w, h, l] : sum of all labels
        squ_zscwhl = np.zeros((len(cfg.obj_types), 6), dtype=np.float64) # sqaure of all label
    


    is_copy_paste = any(d['type_name'] == 'CopyPaste' for d in cfg.data.train_augmentation)
    
    instance_pool = []
    cover_bbox2d_gts = []
    misss_bbox2d_gts = []
    for i, index_name in enumerate(index_names):
        if is_copy_paste:
            
            ######################################
            ### Build copy paste instance_pool ###
            ######################################
            P2   = kitti_calib_file_parser(os.path.join(data_root_dir, "calib", f"{index_name}.txt"))
            with open(os.path.join(data_root_dir, "label_2", f"{index_name}.txt")) as f:
                lines = f.read().splitlines()
                lines = list(lines for lines in lines if lines) # Delete empty lines
            objs =  [CopyPaste_Object(str_line + " NA",
                                      idx_img = index_name,
                                      idx_line = idx_line, 
                                      tf_matrix = P2) for idx_line, str_line in enumerate(lines)]
            # Filter inappropiate objs in instance_pool
            for obj in objs:
                if obj.category in cfg.obj_types and obj.truncated < 0.5 and obj.occluded == 0.0 and obj.area > 3000:
                    if cfg.data.train_augmentation[0]['keywords']['use_seg'] and len(obj.seg_points) == 0 : continue
                    instance_pool.append(obj)
        
        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, image, label, velo, depth = data_frame.read_data()
        
        # Load label , store the list of kittiObjet and kittiCalib, 
        data_frame.label = [obj for obj in label.data if obj.type in cfg.obj_types and obj.occluded < cfg.data.max_occlusion and obj.z > cfg.data.min_z]
        
        if anchor_prior:
            for j in range(len(cfg.obj_types)):
                total_objects[j] += len([obj for obj in data_frame.label if obj.type == cfg.obj_types[j]])
                data = np.array([
                        [obj.z, np.sin(2*obj.alpha), np.cos(2*obj.alpha), obj.w, obj.h, obj.l]
                            for obj in data_frame.label if obj.type==cfg.obj_types[j] ]) #[N, 6]
                if data.any():
                    sum_zscwhl[j, :] += np.sum(data     , axis=0)
                    squ_zscwhl[j, :] += np.sum(data ** 2, axis=0)
        
        # Load calibration file
        data_frame.calib = calib
        if anchor_prior:
            original_image = image.copy()
            
            # Augument the images
            image, P2, label = preprocess(original_image, p2=deepcopy(calib.P2), labels=deepcopy(data_frame.label))
            
            ## Computing statistic for positive anchors
            if len(data_frame.label) > 0:
                anchors, _ = anchor_manager(image[np.newaxis].transpose([0,3,1,2]), torch.tensor(P2).reshape([-1, 3, 4]))
                # print(f"[imdb_precomputer_3d.py] anchors = {anchors.shape}") # [1, 184320, 4]
                
                for j in range(len(cfg.obj_types)):
                    
                    # Label_2d, [num_gt, 4]
                    bbox2d_gt = torch.tensor([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in label if obj.type == cfg.obj_types[j]]).cuda()
                    
                    # Ignore frame with no groundtrues
                    if len(bbox2d_gt) < 1: continue
                    
                    # Label_3d, (x,y,z,sin,cos)
                    bbox3d_gt = torch.tensor([[obj.x, obj.y, obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha)] for obj in label if obj.type == cfg.obj_types[j]]).cuda()
                    
                    usable_anchors = anchors[0] #　[46080, 4]

                    # Get IoU between label and anchors
                    IoUs = calc_iou(usable_anchors, bbox2d_gt) #[N, K], [46080, num_gt]
                    IoU_max       , IoU_argmax        = torch.max(IoUs, dim=0) # IoU_max # [num_gt]
                    IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1) # IoU_max_anchor [46080]
                    # print(f"IoU_max = {IoU_max.shape}") 
                    # print(f"IoU_max_anchor = {IoU_max_anchor.shape}")

                    # Get covered and missed groundtrue for visualization
                    covered_gt_mask = IoU_max > cfg.detector.loss.fg_iou_threshold
                    if bbox2d_gt[ covered_gt_mask].shape[0] != 0: cover_bbox2d_gts.append(bbox2d_gt[ covered_gt_mask])
                    if bbox2d_gt[~covered_gt_mask].shape[0] != 0: misss_bbox2d_gts.append(bbox2d_gt[~covered_gt_mask])
                    
                    total_usable_objects[j] += torch.sum(covered_gt_mask).item()

                    positive_anchors_mask = IoU_max_anchor > cfg.detector.loss.fg_iou_threshold
                    positive_gts_xyzsc = bbox3d_gt[IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()
                    # print(f"positive_gts_xyzsc = {positive_gts_xyzsc.shape}") # (num_positive, 5)

                    if external_anchor_path == "":
                        used_anchors = usable_anchors[positive_anchors_mask].cpu().numpy() #[x1, y1, x2, y2]
                        # print(f"used_anchors = {used_anchors.shape}")
                        sizes_int, ratio_int = anchor_manager.anchors2indexes(used_anchors)
                        for k in range(len(sizes_int)):
                            num_covered_gt[j, sizes_int[k], ratio_int[k]] += 1 # Denominator, number of groundtrue cover by this anchor
                            sum_covered_gt[j, sizes_int[k], ratio_int[k]] += positive_gts_xyzsc[k, 2:5]       # [z, sin, cos]
                            squ_covered_gt[j, sizes_int[k], ratio_int[k]] += positive_gts_xyzsc[k, 2:5] ** 2  # [z, sin, cos]
                    else:
                        # get the indices of the True values in positive_anchors_mask
                        anchor_idx = np.where(positive_anchors_mask.cpu().numpy())[0]
                        ank_i = anchor_idx % num_external_anchor
                        num_covered_gt[j, ank_i] += 1 # Denominator, number of groundtrue cover by this anchor
                        sum_covered_gt[j, ank_i] += positive_gts_xyzsc[:, 2:5]       # [z, sin, cos]
                        squ_covered_gt[j, ank_i] += positive_gts_xyzsc[:, 2:5] ** 2  # [z, sin, cos]
                    
        # Save label.txt and calib.txt in frames[ KittiData(), KittiData(), KittiData(), ...]
        frames[i] = data_frame

        if (i+1) % time_display_inter == 0:
            avg_time = timer.compute_avg_time(i+1)
            eta = timer.compute_eta(i+1, N)
            print("{} iter:{}/{}, avg-time:{}, eta:{}, total_objs:{}, usable_objs:{}".format(
                "training", i+1, N, avg_time, eta, total_objects, total_usable_objects), end='\r')
    
    print("\n")
    print(f"Best Possible Recall = {100*total_usable_objects[0]/total_objects[0]}%")
    
    save_dir = os.path.join(cfg.path.preprocessed_path, "training")
    if is_copy_paste:
        # print(AUGMENTATION_DICT.get('CopyPaste').load_path())
        AUGMENTATION_DICT.get('CopyPaste').load_path(os.path.join(save_dir, "instance_pool.pkl"), os.path.join(save_dir, "imgs_src.pkl"))
    
    ####################################
    ### Anchor mean, std Calculation ###
    ####################################
    if anchor_prior:
        for j in range(len(cfg.obj_types)):
            
            if external_anchor_path == "":
                avg = sum_covered_gt[j] / (num_covered_gt[j][:, :, np.newaxis] + 1e-8)
                ex2 = squ_covered_gt[j] / (num_covered_gt[j][:, :, np.newaxis] + 1e-8)
                std = np.sqrt(ex2 - avg ** 2)
            else:
                avg = sum_covered_gt[j] / (num_covered_gt[j][:, np.newaxis] + 1e-8) # (32, 3)
                ex2 = squ_covered_gt[j] / (num_covered_gt[j][:, np.newaxis] + 1e-8) 
                std = np.sqrt(ex2 - avg ** 2) # (32, 3)
            
            # If the covered groundtrue is less than 10, ignore the avg and std
            avg[num_covered_gt[j] < 10, :] = -100  # with such negative mean Z, anchors/losses will filter them out
            std[num_covered_gt[j] < 10, :] = 1e10
            
            avg[np.isnan(std)]      = -100
            std[np.isnan(std)]      = 1e10
            avg[std < 1e-3]         = -100
            std[std < 1e-3]         = 1e10
            
            # Get width, height, length statistics
            # It is the average of the whole groundtrue
            global_mean = sum_zscwhl[j] / total_objects[j] # (6,)
            global_var  = np.sqrt(squ_zscwhl[j] / total_objects[j] - global_mean**2) # (6,)
            
            if external_anchor_path == "":
                whl_avg = np.ones([avg.shape[0], avg.shape[1], 3]) * global_mean[3:6]
                whl_std = np.ones([avg.shape[0], avg.shape[1], 3]) * global_var [3:6]
                avg = np.concatenate([avg, whl_avg], axis=2)
                std = np.concatenate([std, whl_std], axis=2)
            else:
                whl_avg = np.ones([num_external_anchor, 3]) * global_mean[3:6]
                whl_std = np.ones([num_external_anchor, 3]) * global_var [3:6]
                avg = np.concatenate([avg, whl_avg], axis=1) # (32, 6)
                std = np.concatenate([std, whl_std], axis=1) # (32, 6)
            
            # Output convered and missed groundtrue for visualization
            anchor_prior_calculation_result = {}
            anchor_prior_calculation_result['cover_bbox2d_gts'] = torch.cat(cover_bbox2d_gts, dim=0).cpu().numpy()
            anchor_prior_calculation_result['misss_bbox2d_gts'] = torch.cat(misss_bbox2d_gts, dim=0).cpu().numpy()
            anchor_prior_calculation_result['anchor_all_bbox2d'] = anchors[0].cpu().numpy()
            anchor_prior_calculation_result['anchor_bbox2d']    = bbox2d
            anchor_prior_calculation_result['anchor_mean']      = avg # (32, 6)
            anchor_prior_calculation_result['anchor_std']       = std # (32, 6)
            
            # with open('/home/lab530/KenYu/visualDet3D/covered_missed_gt/anchor_prior_calculation_result.pkl', 'wb') as f:
            #     pickle.dump(anchor_prior_calculation_result, f)
            #     print(f"[imdb_precompute_3d.py] Saved anchor_prior_calculation_result")
            
            # Save anchor file for later training 
            np.save(os.path.join(save_dir,'anchor_mean_{}.npy'.format(cfg.obj_types[j])), avg)
            np.save(os.path.join(save_dir,'anchor_std_{}.npy' .format(cfg.obj_types[j])), std)

    ######################################
    ### Save Scene-Aware Instance Pool ###
    ######################################
    if is_copy_paste:
        
        # Save image source
        imgs_src  = {fn.split(".")[0]: cv2.imread(os.path.join(data_root_dir, "image_2", fn)) for fn in os.listdir(os.path.join(data_root_dir, "image_2"))}
        print(f"Number of source image in imgs_src: {len(imgs_src)}")
        pickle.dump(imgs_src, open(os.path.join(save_dir, "imgs_src.pkl"), "wb"))
        print(f"Saved source images to {os.path.join(save_dir, 'imgs_src.pkl')}")
        
        # Save instance pool
        print(f"Number of object in instance pool: {len(instance_pool)}")
        pickle.dump(instance_pool, open(os.path.join(save_dir, "instance_pool.pkl"), "wb"))
        print(f"Saved instance pool to {os.path.join(save_dir, 'instance_pool.pkl')}")
        
    print("training split finished precomputing")
    return frames

def preprocess_test_dataset(cfg, index_names, data_root_dir, output_dict):

    frames = [None] * len(index_names)
    print("start reading testing data")

    for i, index_name in enumerate(index_names):

        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, _, _, _, _ = data_frame.read_data()

        # store the list of kittiObjet and kittiCalib
        data_frame.calib = calib

        frames[i] = data_frame

    print("test split finished precomputing")
    return frames

def preprocess_test_sequence_dataset(cfg, index_names, data_root_dir, output_dict):

    frames = [None] * len(index_names)
    print("start reading testing data")

    for i, index_name in enumerate(index_names):

        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        data_frame.calib_path  = os.path.join(data_root_dir, "calib"   , f"{index_name.split('/')[0]}.txt")
        data_frame.image2_path = os.path.join(data_root_dir, "image_02", index_name + '.png')

        calib, _, _, _, _ = data_frame.read_data()

        # store the list of kittiObjet and kittiCalib
        data_frame.calib = calib

        frames[i] = data_frame

    print("test split finished precomputing")
    return frames