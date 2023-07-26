"""
    Script for launching training process
"""
import matplotlib
matplotlib.use('agg') 

import os
from easydict import EasyDict
from fire import Fire
import torch
from torch.utils.tensorboard import SummaryWriter

from _path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from visualDet3D.evaluator.kitti.evaluate import evaluate
import visualDet3D.data.kitti.dataset
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import LossLogger, cfg_from_file
from visualDet3D.networks.optimizers import optimizers, schedulers
from visualDet3D.data.dataloader import build_dataloader

def main(config="config/config.py", experiment_name="default", world_size=1, local_rank=-1):
    """Main function for the training script.

    KeywordArgs:
        config (str): Path to config file.
        experiment_name (str): Custom name for the experitment, only used in tensorboard.
        world_size (int): Number of total subprocesses in distributed training. 
        local_rank: Rank of the process. Should not be manually assigned. 0-N for ranks in distributed training (only process 0 will print info and perform testing). -1 for single training. 
    """

    ## Get config
    cfg = cfg_from_file(config)

    ## Collect distributed(or not) information
    cfg.dist = EasyDict()
    cfg.dist.world_size = world_size
    cfg.dist.local_rank = local_rank
    is_distributed = local_rank >= 0 # local_rank < 0 -> single training
    is_logging     = local_rank <= 0 # only log and test with main process
    is_evaluating  = local_rank <= 0

    ## Setup writer if local_rank > 0
    # recorder_dir = os.path.join(cfg.path.log_path, experiment_name + f"config={config}")
    recorder_dir = os.path.join(cfg.path.log_path, experiment_name)
    print(f"is_logging = {is_logging}") # True

    if is_logging: # writer exists only if not distributed and local rank is smaller
        ## Clean up the dir if it exists before
        if os.path.isdir(recorder_dir):
            os.system("rm -r {}".format(recorder_dir))
            print("clean up the recorder directory of {}".format(recorder_dir))
        
        writer = SummaryWriter(recorder_dir)
        # writer.add_custom_scalars(layout)

        ## Record config object using pprint
        import pprint
        formatted_cfg = pprint.pformat(cfg)
        writer.add_text("config.py", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n')) # add space for markdown style in tensorboard text
    else:
        writer = None

    ## Set up GPU and distribution process
    if is_distributed:
        cfg.trainer.gpu = local_rank # local_rank will overwrite the GPU in configure file
    gpu = min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(cfg.trainer, 'cudnn', False)
    torch.cuda.set_device(gpu)
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print(local_rank) # -1
 
    ## define datasets and dataloader.
    dataset_train = DATASET_DICT[cfg.data.train_dataset](cfg)
    dataset_val = DATASET_DICT[cfg.data.val_dataset](cfg, "validation")
    # for i in dataset_train:
    #     print(i.keys()) # ['calib', 'image', 'label', 'bbox2d', 'bbox3d', 'original_shape', 'original_P', 'loc_3d_roy']
    print('Num training images: {}'.format(len(dataset_train)))
    print('Num validation images: {}'.format(len(dataset_val)))
    dataloader_train = build_dataloader(dataset_train,
                                        num_workers=cfg.data.num_workers,
                                        batch_size=cfg.data.batch_size,
                                        collate_fn=dataset_train.collate_fn,
                                        local_rank=local_rank,
                                        world_size=world_size,
                                        sampler_cfg=getattr(cfg.data, 'sampler', dict()))

    #
    print(f"Experiment Setting: {cfg.exp}")

    # Set default value for iou_type
    cfg.detector.loss.iou_type = getattr(cfg.detector.loss, 'iou_type', 'baseline')
    print(f"iou_type = {cfg.detector.loss.iou_type}")
    
    ## Create the model
    detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
    # print(detector)

    ## Load old model if needed
    old_checkpoint = getattr(cfg.path, 'pretrained_checkpoint', None)
    print(f"old_checkpoint = {old_checkpoint}") # None
    
    if old_checkpoint is not None:
        # state_dict = torch.load(old_checkpoint, map_location='cpu')
        # detector.load_state_dict(state_dict)

        print(f"Use pretrained model at {old_checkpoint}")
        checkpoint = torch.load(old_checkpoint, map_location='cpu')

        # Load partial of the pre-train model
        pretrained_dict = checkpoint # ['model_state_dict']
        model_dict = detector.state_dict()
        
        # Reference: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        # 1. filter out unnecessary keys
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                filtered_dict[k] = v
            else:
                print(f"Filtered in pretrain weight {k}")
        pretrained_dict = filtered_dict
        
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(f"pretrained_dict = {pretrained_dict.keys()}")
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        detector.load_state_dict(model_dict)

    ## Convert to cuda
    if is_distributed:
        detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector)
        detector = torch.nn.parallel.DistributedDataParallel(detector.cuda(), device_ids=[gpu], output_device=gpu)
    else:
        detector = detector.cuda()
    detector.train()

    ## Record basic information of the model
    if is_logging:
        string1 = detector.__str__().replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text("model structure", string1) # add space for markdown style in tensorboard text
        num_parameters = get_num_parameters(detector)
        print(f'number of trained parameters of the model: {num_parameters}')
    
    ## define optimizer and weight decay
    optimizer = optimizers.build_optimizer(cfg.optimizer, detector)

    ## define scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.trainer.max_epochs, cfg.optimizer.lr_target)
    scheduler_config = getattr(cfg, 'scheduler', None)
    scheduler = schedulers.build_scheduler(scheduler_config, optimizer)
    is_iter_based = getattr(scheduler_config, "is_iter_based", False)

    ## define loss logger
    training_loss_logger =  LossLogger(writer, 'train') if is_logging else None

    ## training pipeline
    if 'training_func' in cfg.trainer:
        training_dection = PIPELINE_DICT[cfg.trainer.training_func]
    else:
        raise KeyError

    ## Get evaluation pipeline
    if 'evaluate_func' in cfg.trainer:
        evaluate_detection = PIPELINE_DICT[cfg.trainer.evaluate_func]
        print("Found evaluate function {}".format(cfg.trainer.evaluate_func))
    else:
        evaluate_detection = None
        print("Evaluate function not found")


    ## timer is used to estimate eta
    timer = Timer()

    # Record the current best validation evaluation result
    best_result = (-1, -1, -1, -1, -1, -1, -1, -1, -1) # 3d_e, 3d_m, 3d_h, bev_e, bev_m, bev_h, bbox_e, bbox_m, bbox_h
    best_epoch  = -1
    
    global_step = 0
    for epoch_num in range(cfg.trainer.max_epochs):
        ## Start training for one epoch
        detector.train()
        if training_loss_logger:
            training_loss_logger.reset()
        for iter_num, data in enumerate(dataloader_train):
            loss_dict = training_dection(data, detector, optimizer, writer, training_loss_logger, global_step, epoch_num, cfg)
                        
            global_step += 1

            if is_iter_based:
                scheduler.step()

            if is_logging and global_step % cfg.trainer.disp_iter == 0:
                ## Log loss, print out and write to tensorboard in main process
                if '1/total_loss' not in training_loss_logger.loss_stats:
                    print(f"\nIn epoch {epoch_num}, iteration:{iter_num}, global_step:{global_step}, total_loss not found in logger.")
                else:
                    log_str = 'Epoch: {} | Iteration: {}  | Running loss: {:1.5f} | eta:{}'.format(
                        epoch_num, iter_num, training_loss_logger.loss_stats['1/total_loss'].avg,
                        timer.compute_eta(global_step, len(dataloader_train) * cfg.trainer.max_epochs))
                    print(log_str, end='\r')
                    writer.add_text("training_log/train", log_str, global_step)
                    training_loss_logger.log(global_step)

        if not is_iter_based:
            scheduler.step()

        ## save model in main process if needed
        if is_logging:
            torch.save(detector.module.state_dict() if is_distributed else detector.state_dict(), os.path.join(
                cfg.path.checkpoint_path, '{}_latest.pth'.format(
                    cfg.detector.name)
                )
            )
        if is_logging and (epoch_num + 1) % cfg.trainer.save_iter == 0:
            torch.save(detector.module.state_dict() if is_distributed else detector.state_dict(), os.path.join(
                cfg.path.checkpoint_path, '{}_{}.pth'.format(
                    cfg.detector.name,epoch_num)
                )
            )

        ## test model in main process if needed
        if is_evaluating and evaluate_detection is not None and cfg.trainer.test_iter > 0 and (epoch_num + 1) % cfg.trainer.test_iter == 0:
            print("\n/**** start testing after training epoch {} ******/".format(epoch_num))
            eval_result = evaluate_detection(cfg, detector.module if is_distributed else detector, dataset_val, writer, epoch_num)
            print("/**** finish testing after training epoch {} ******/".format(epoch_num))
            
            # check whether is the best run, don't consider 2d bbox result
            if sum(eval_result[:6])/len(eval_result[:6]) > sum(best_result[:6])/len(best_result[:6]):
                best_result = eval_result
                best_epoch  = epoch_num
                print("This is the new best validation result")
                
        if is_distributed:
            torch.distributed.barrier() # wait untill all finish a epoch

        if is_logging:
            writer.flush()
    
    print(f"Train finish, the best_epoch is {best_epoch}")
    print(f"Best validation evaluation result:")
    print(f"3d_e, 3d_m, 3d_h, bev_e, bev_m, bev_h, bbox_e, bbox_m, bbox_h: {best_result}")
    writer.add_text("best validation result", str(best_result))
    writer.flush()

if __name__ == '__main__':
    Fire(main)
