import os
from fire import Fire
from easydict import EasyDict as edict
import torch
import pprint
import numpy as np

# Reference: https://blog.csdn.net/xr627/article/details/127581608
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # For avoid tensorflow mumbling

import sys
visualDet3D_path = os.path.dirname(sys.path[0])  #two folders upwards
sys.path.insert(0, visualDet3D_path)

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from visualDet3D.data.kitti_dataset import KittiDataset
from visualDet3D.data.sampler import TrainingSampler
from visualDet3D.data.preprocess import preprocess_train_dataset, process_train_val_file, preprocess_val_dataset
from visualDet3D.utils.cal import get_num_parameters
from visualDet3D.networks.optimizers import optimizers, schedulers
from visualDet3D.networks.detectors.yolo3d_detector import Yolo3D
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import LossLogger, cfg_from_file
from visualDet3D.utils.utils import compound_annotation
from visualDet3D.evaluator.evaluators import evaluate_kitti_obj

# For print out loss
loss_avg_dict = {"1/reg_loss": 0,
                 "1/cls_loss": 0,
                 "1/dep_loss": 0,
                 "2/dx": 0,
                 "2/dy": 0,
                 "2/dw": 0,
                 "2/dh": 0,
                 "2/cdx": 0,
                 "2/cdy": 0,
                 "2/cdz": 0,
                 "4/dw": 0,
                 "4/dh": 0,
                 "4/dl": 0,}

def main(cfg_path="config/project_name/exp_name.py", world_size=1, local_rank=-1):
    """
    KeywordArgs:
        cfg_path (str): Path to config file.
        world_size (int): Number of total subprocesses in distributed training. 
        local_rank: Rank of the process. Should not be manually assigned. 0-N for ranks in distributed training (only process 0 will print info and perform testing). -1 for single training. 
    """

    assert len(cfg_path.split('/')) == 3, "config_path must be in the format of config/project_name/exp_name.py" 
    
    cfg = cfg_from_file(cfg_path)


    torch.cuda.set_device(cfg.trainer.gpu)
    
    is_copy_paste = any(d['type_name'] == 'CopyPaste' for d in cfg.data.train_augmentation)
    
    # Collect distributed(or not) information
    cfg.dist = edict()
    cfg.dist.world_size = world_size
    cfg.dist.local_rank = local_rank
    is_distributed = local_rank >= 0 # local_rank < 0 -> single training
    is_logging     = local_rank <= 0 # only log and test with main process
    is_evaluating  = local_rank <= 0

    if is_logging: # writer exists only if not distributed and local rank is smaller
        # Clean up the dir if it exists before
        if os.path.isdir(cfg.path.log_path):
            os.system("rm -r {}".format(cfg.path.log_path))
            print("clean up the recorder directory of {}".format(cfg.path.log_path))
        
        writer = SummaryWriter(cfg.path.log_path)

        # Record config object using pprint
        formatted_cfg = pprint.pformat(cfg)
        writer.add_text("config.py", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n')) # add space for markdown style in tensorboard text
    else:
        writer = None

    ## Set up GPU and distribution process
    # Setup writer if local_rank > 0
    if is_distributed:
        cfg.trainer.gpu = local_rank # local_rank will overwrite the GPU in configure file
    gpu = min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(cfg.trainer, 'cudnn', False)
    torch.cuda.set_device(gpu)
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Load training datatset
    train_names, val_names = process_train_val_file(cfg)
    output_dict = {"calib": True,
                   "image": True,
                   "label": True,
                   "velodyne": False,
                   "depth": is_copy_paste,}
    imdb_frames_train = preprocess_train_dataset(cfg, train_names, cfg.data.train_data_path, output_dict)
    
    # Load validation datatset
    output_dict = {"calib": True,
                   "image": False,
                   "label": True,
                   "velodyne": False,
                   "depth": False,}
    imdb_frames_val = preprocess_val_dataset(cfg, val_names, cfg.data.train_data_path, output_dict)

    print("Preprocessing finished")

    # define datasets and dataloader.
    dataset_train = KittiDataset(cfg, imdb_frames_train, "training")
    dataset_val   = KittiDataset(cfg, imdb_frames_val, "validation")
    print('[train.py] Number of training images: {}'.format(len(dataset_train)))
    print('[train.py] Number of validation images: {}'.format(len(dataset_val)))

    dataloader_train = DataLoader(dataset_train, 
                        num_workers = cfg.data.num_workers, 
                        batch_size  = cfg.data.batch_size, 
                        collate_fn  = dataset_train.collate_fn, 
                        sampler = TrainingSampler(size=len(dataset_train), rank=local_rank, world_size=world_size,), 
                        drop_last=True)

    # Create the model
    detector = Yolo3D(cfg)

    # Load pretrain model
    if cfg.ckpt_path != "":
        print(f"[train.py] Use pretrained model at {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location='cpu')
        
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
        
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # 3. load the new state dict
        detector.load_state_dict(model_dict)

    # Convert to cuda
    if is_distributed:
        detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector)
        detector = torch.nn.parallel.DistributedDataParallel(detector.cuda(), device_ids=[gpu], output_device=gpu)
    else:
        detector = detector.cuda()
    detector.train()

    # Record basic information of the model
    if is_logging:
        string1 = detector.__str__().replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text("model structure", string1) # add space for markdown style in tensorboard text
        num_parameters = get_num_parameters(detector)
        print(f'[train.py] number of trained parameters of the model: {num_parameters}')
    
    # define optimizer and weight decay
    optimizer = optimizers.build_optimizer(cfg.optimizer, detector)

    # define scheduler
    scheduler_config = getattr(cfg, 'scheduler', None)
    scheduler = schedulers.build_scheduler(scheduler_config, optimizer)
    is_iter_based = getattr(scheduler_config, "is_iter_based", False)

    # define loss logger
    training_loss_logger = LossLogger(writer, 'train') if is_logging else None

    # timer is used to estimate eta
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
            # loss_dict = training_dection(data, detector, optimizer, writer, training_loss_logger, global_step, epoch_num, cfg)

            optimizer.zero_grad()
            # load data
            image, calibs, labels, bbox2d, bbox_3d, loc_3d_ry = data
            
            # create compound array of annotation
            max_length = np.max([len(label) for label in labels])
            if max_length == 0:
                return

            print_loss   = getattr(cfg, 'print_loss', False)
            annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, loc_3d_ry, cfg.obj_types, calibs) #np.arraym, [batch, max_length, 4 + 1 + 7]

            loss_dict = detector([image.cuda().contiguous(), image.new(annotation).cuda(), calibs.cuda()])

            # Print out loss on screen
            if print_loss:
                if global_step % 10 == 0:
                    print("Step {:6} | reg_loss:{:.5f} cls_loss:{:.5f} | dx:{:.5f} dy:{:.5f} dw:{:.5f} dh:{:.5f} | cx:{:.5f} cy:{:.5f} cz:{:.5f} | dw:{:.5f} dh:{:.5f} dl:{:.5f} |  ".format(global_step, 
                            loss_dict['1/reg_loss'].detach().cpu().numpy()[0],
                            loss_dict['1/cls_loss'].detach().cpu().numpy()[0],
                            loss_dict['2/dx'],
                            loss_dict['2/dy'],
                            loss_dict['2/dw'],
                            loss_dict['2/dh'],
                            loss_dict['2/cdx'],
                            loss_dict['2/cdy'],
                            loss_dict['2/cdz'],
                            loss_dict['4/dw'],
                            loss_dict['4/dh'],
                            loss_dict['4/dl']))
                    # Reset loss_avg_dict
                    for k in loss_avg_dict: loss_avg_dict[k] = 0
                else:
                    for k in loss_avg_dict:
                        try:
                            if k == '1/reg_loss' or k == '1/cls_loss' or k == '1/dep_loss':
                                loss_avg_dict[k] += loss_dict[k].detach().cpu().numpy()[0] / 10
                            else:
                                loss_avg_dict[k] += loss_dict[k] / 10
                        except KeyError:
                            pass
            
            # Record loss in a average meter
            if training_loss_logger is not None:
                training_loss_logger.update(loss_dict)
            loss = loss_dict['1/total_loss']
            
            if bool(loss.item() == 0):
                return
            
            # Back probagation
            loss.backward()

            # Clip loss norm to avoid overfitting
            torch.nn.utils.clip_grad_norm_(detector.parameters(), cfg.optimizer.clipped_gradient_norm)

            optimizer.step()
            optimizer.zero_grad()

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

        # save model in main process if needed
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

        if is_evaluating and cfg.trainer.test_iter > 0 and (epoch_num + 1) % cfg.trainer.test_iter == 0:
            print("\n/**** start testing after training epoch {} ******/".format(epoch_num))
            eval_result = evaluate_kitti_obj(cfg, detector.module if is_distributed else detector, dataset_val, writer, epoch_num, output_path = os.path.join(cfg.path.preprocessed_path, "validation", "data"))
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
    print(f"AP_3D_easy, AP_3D_moderate, AP_3D_hard, AP_BEV_easy, AP_BEV_moderate, AP_BEV_hard, AP_2D_easy, AP_2D_moderate, AP_2D_hard: {best_result}")
    writer.add_text("best validation result", str(best_result))
    writer.flush()

if __name__ == '__main__':
    Fire(main)
