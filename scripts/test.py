import matplotlib # Disable GUI
matplotlib.use('agg')
import os
import time
import fire
import torch
from tqdm import tqdm
import sys

visualDet3D_path = os.path.dirname(sys.path[0])  #two folders upwards
sys.path.insert(0, visualDet3D_path)

from visualDet3D.networks.detectors.yolo3d_detector import Yolo3D
from visualDet3D.utils.utils import cfg_from_file
from visualDet3D.evaluator.evaluators import evaluate_kitti_obj, test_one
from visualDet3D.data.kitti_dataset import KittiDataset, KittiTestDataset
from visualDet3D.data.preprocess import preprocess_train_dataset, process_train_val_file, preprocess_test_dataset, preprocess_val_dataset
from visualDet3D.utils.cal import BBox3dProjector, BackProjection

def main(cfg_path:str="config/project_name/exp_name.py",
         gpu:int=0, 
         checkpoint_path:str="",
         split_to_test:str='val',
         output_path:str=""):
    
    assert len(cfg_path.split('/')) == 3, "config_path must be in the format of config/project_name/exp_name.py" 
    
    if checkpoint_path == "":
        checkpoint_path = os.path.join("exp_output", cfg_path.split('/')[1], cfg_path.split('/')[2], "checkpoint", "Yolo3D_latest.pth")

    # Read Config
    cfg = cfg_from_file(cfg_path)

    # Force GPU selection in command line
    cfg.trainer.gpu = gpu
    torch.cuda.set_device(cfg.trainer.gpu)

    if split_to_test == 'train':
        train_names, val_names = process_train_val_file(cfg)
        output_dict = {
                    "calib": True,
                    "image": True,
                    "label": True,
                    "velodyne": False,
                    "depth": False,}
        imdb_frames_train = preprocess_train_dataset(cfg, train_names, cfg.data.train_data_path, output_dict)
        dataset = KittiDataset(cfg, imdb_frames_train, "training")
    
    elif split_to_test == 'val':
        train_names, val_names = process_train_val_file(cfg)
        output_dict = {
                    "calib": True,
                    "image": False,
                    "label": True,
                    "velodyne": False,
                    "depth": False,}
        imdb_frames_val = preprocess_val_dataset(cfg, val_names, cfg.data.train_data_path, output_dict)
        dataset = KittiDataset(cfg, imdb_frames_val, "validation")
    
    elif split_to_test == 'test':
        list_calib = os.listdir( os.path.join(cfg.data.test_data_path, "calib") )
        test_names = [i.split('.')[0] for i in list_calib]
        output_dict = {
                    "calib": True,
                    "image": False,
                    "label": False,
                    "velodyne": False,
                    "depth": False,}
        imdb_frames_test = preprocess_test_dataset(cfg, test_names, cfg.data.test_data_path, output_dict)
        dataset = KittiTestDataset(cfg, imdb_frames_test, "test")
    
    else:
        raise NotImplementedError

    print(f"Number of image in dataset: {len(dataset)}")

    # Create 3D object detector
    detector = Yolo3D(cfg).cuda()
    
    # Load pretrain model
    state_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
    detector.load_state_dict(state_dict.copy(), strict=False)
    detector.eval()

    # Clean the result path
    if output_path == "":
        if split_to_test == "train":
            output_path = os.path.join(cfg.path.preprocessed_path, "training", "data")
        elif split_to_test == "val":
            output_path = os.path.join(cfg.path.preprocessed_path, "validation", "data")
        elif split_to_test == "test":
            output_path = os.path.join(cfg.path.preprocessed_path, "testing", "data")
    
    if os.path.isdir(output_path):
        os.system("rm -r {}".format(output_path))
    os.mkdir(output_path)
    print("Clean up the recorder directory of {}".format(output_path))

    t_start = time.time()

    # Evaluation or Inference
    if split_to_test == "train" or split_to_test == "val":
        eval_result = evaluate_kitti_obj(cfg, detector, dataset, None, 0, output_path = output_path)
    else:
        # Only inference on testing dataset
        fn_list = [i.split('.')[0] for i in os.listdir( os.path.join(cfg.data.test_data_path, "image_2"))]
        
        assert len(fn_list) == len(dataset), f'Number of validation data are not matched. fn_list has {len(fn_list)} files, but dataset_val has {len(dataset_val)} files'
        
        projector     = BBox3dProjector().cuda()
        backprojector = BackProjection().cuda()

        for index in tqdm(range(len(dataset))):
            test_one(cfg, index, fn_list, dataset, detector, backprojector, projector, output_path)
    
    t_spend = time.time() - t_start
    
    print(f"Total time used: {t_spend}")
    print(f"FPS = {1 / (t_spend/len(dataset))}")

if __name__ == '__main__':
    fire.Fire(main)
