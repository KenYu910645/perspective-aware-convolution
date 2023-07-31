from easydict import EasyDict as edict
import os 
import numpy as np

cfg = edict()
cfg.obj_types = ['Car']

cfg.trainer = edict(
    gpu = 0,
    max_epochs = 30,
    disp_iter = 1,
    save_iter = 99,
    test_iter = 1,
    training_func = "train_mono_detection",
    test_func = "test_mono_detection",
    evaluate_func = "evaluate_kitti_obj",
)

# path
# path = edict()
cfg.path.data_path = 'dataset/kitti/training'# "/data/kitti_obj/training" # used in visualDet3D/data/.../dataset
cfg.path.test_path = 'dataset/kitti/testing' # ""
# path.visualDet3D_path = 'visualDet3D' # "/path/to/visualDet3D/visualDet3D" # The path should point to the inner subfolder
# path.project_path = 'exp_output/pac_new' # "/path/to/visualDet3D/workdirs" # or other path for pickle files, checkpoints, tensorboard logging and output files.
# path.pretrained_checkpoint = "/home/lab530/KenYu/visualDet3D/exp_output/mixup/kitti_mixup_1/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth"
# cfg.path = path

## optimizer
cfg.optimizer = edict(
    type_name = 'adam',
    keywords = edict(lr = 1e-4, weight_decay = 0,),
    clipped_gradient_norm = 0.1
)
# cfg.optimizer = optimizer

cfg.scheduler = edict(
    type_name = 'CosineAnnealingLR',
    keywords = edict(T_max = cfg.trainer.max_epochs, eta_min = 3e-5,),
)
# cfg.scheduler = scheduler

cfg.data = edict(
    batch_size = 8,
    num_workers = 8,
    rgb_shape = (288, 1280, 3),
    train_dataset = "KittiMonoDataset",
    val_dataset   = "KittiMonoDataset",
    test_dataset  = "KittiMonoTestDataset",
    train_split_file = os.path.join('visualDet3D/data/kitti/kitti_anchor_gen_split/train_all.txt'),
    val_split_file   = os.path.join('visualDet3D/data/kitti/kitti_anchor_gen_split/val_all.txt'),
    use_right_image = False,
    max_occlusion =  1000, # 2
    min_z         = -1000, # 3
    is_overwrite_anchor_file = False,
)

cfg.data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    crop_top = 100,
)

cfg.data.train_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='PhotometricDistort', keywords=edict(distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32)),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='RandomMirror', keywords=edict(mirror_prob=0.5)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
cfg.data.test_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
# cfg.data = data

## networks
# detector = edict()
cfg.detector.obj_types = cfg.obj_types
cfg.detector.name = 'Yolo3D'
cfg.detector.backbone = edict(
    depth=101,
    pretrained=True,
    frozen_stages=-1,
    num_stages=3,
    out_indices=(2, ),
    norm_eval=False,
    dilations=(1, 1, 1),
)

# head_loss = edict(
#     fg_iou_threshold = 0.5,
#     bg_iou_threshold = 0.4,
#     L1_regression_alpha = 5 ** 2,
#     focal_loss_gamma = 2.0,
#     match_low_quality=False,
#     balance_weight   = [20.0],
#     regression_weight = [1, 1, 1, 1, 1, 1, 3, 1, 1, 0.5, 0.5, 0.5, 1], #[x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
#     filter_anchor = False, # This prevent anchor filtering !!
# )

# head_test = edict(
cfg.detector.test = edict(
    score_thr=0.5, # TODO, 0.75
    cls_agnostic = False,
    nms_iou_thr=0.5, # TODO  , 0.5, bigger -> striker
    post_optimization = False, # TODO, True
)

# anchors = edict(
#         {
#             'obj_types': cfg.obj_types,
#             'pyramid_levels':[4],
#             'strides': [2 ** 4],
#             'sizes' : [24],
#             'ratios': np.array([0.5, 1]),
#             'scales': np.array([2 ** (i / 4.0) for i in range(16)]),
#         }
#     )

# head_layer = edict(
#     num_features_in=1024,
#     num_anchors=32,
#     num_cls_output=len(cfg.obj_types)+1,
#     num_reg_output=12,
#     cls_feature_size=512,
#     reg_feature_size=1024,
# )

cfg.detector.head = edict(
    num_regression_loss_terms=13,
    num_classes     = len(cfg.obj_types),
    # anchors_cfg     = anchors,
    # layer_cfg       = head_layer,
    # loss_cfg        = head_loss,
    # test_cfg        = head_test,
    # data_cfg        = data,
    is_pac_module   = True,
    num_features_in=1024,
    num_anchors=32,
    num_cls_output=len(cfg.obj_types)+1,
    num_reg_output=12,
    cls_feature_size=512,
    reg_feature_size=1024,
)

cfg.detector.anchors = edict(
    obj_types = cfg.obj_types,
    pyramid_levels = [4],
    strides = [2 ** 4],
    sizes = [24],
    ratios = np.array([0.5, 1]),
    scales = np.array([2 ** (i / 4.0) for i in range(16)]),
)

cfg.detector.loss = edict(
    fg_iou_threshold = 0.5,
    bg_iou_threshold = 0.4,
    L1_regression_alpha = 5 ** 2,
    focal_loss_gamma = 2.0,
    match_low_quality=False,
    balance_weight   = [20.0],
    regression_weight = [1, 1, 1, 1, 1, 1, 3, 1, 1, 0.5, 0.5, 0.5, 1], #[x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
    filter_anchor = False, # This prevent anchor filtering !!
)

# cfg.detector = detector
