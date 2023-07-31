"""
    This script contains function snippets for different training settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from visualDet3D.utils.utils import LossLogger
from visualDet3D.utils.utils import compound_annotation
from visualDet3D.networks.utils.registry import PIPELINE_DICT

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

@PIPELINE_DICT.register_module
def train_mono_detection(data, module:nn.Module,
                     optimizer:optim.Optimizer,
                     writer:SummaryWriter=None, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     epoch_num:int=None,
                     cfg:EasyDict=EasyDict()):
    optimizer.zero_grad()
    # load data
    image, calibs, labels, bbox2d, bbox_3d, loc_3d_ry = data
    
    # create compound array of annotation
    max_length = np.max([len(label) for label in labels])
    if max_length == 0:
        return

    is_noam_loss = getattr(cfg.detector.head, 'is_noam_loss', False)
    print_loss   = getattr(cfg              , 'print_loss'  , False)
    annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, loc_3d_ry, cfg.obj_types, calibs, is_noam_loss = is_noam_loss) #np.arraym, [batch, max_length, 4 + 1 + 7]

    # Feed to the network
    # classification_loss, regression_loss, loss_dict = module(
    #         [image.cuda().contiguous(), image.new(annotation).cuda(), calibs.cuda()])
    # classification_loss = classification_loss.mean()
    # regression_loss = regression_loss.mean()

    loss_dict = module([image.cuda().contiguous(), image.new(annotation).cuda(), calibs.cuda()])

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
    if loss_logger is not None:
        loss_logger.update(loss_dict)
    loss = loss_dict['1/total_loss']
    
    if bool(loss.item() == 0):
        return
    loss.backward()
    # clip loss norm
    torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm)

    optimizer.step()
    optimizer.zero_grad()
