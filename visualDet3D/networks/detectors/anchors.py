from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import os
import pickle

def load_from_pkl_or_npy(file_path):
    if   file_path.split(".")[-1] == "npy":
        print(f"[anchors.py] Load npy file from {file_path}")
        return np.load(file_path)
    
    elif file_path.split(".")[-1] == "pkl":
        print(f"[anchors.py] Load pkl file from {file_path}")
        with open(file_path, 'rb') as f: return pickle.load(f)
    
    else:
        raise NotImplementedError

class Anchors(nn.Module):
    def __init__(self, cfg, is_training_process = True):
        super(Anchors, self).__init__()

        self.cfg = cfg
        self.anchor_cfg = cfg.detector.anchors
        self.is_training_process = is_training_process
        print(f"[anchors.py] self.is_training_process = {self.is_training_process}") # True during training, False during preprocessing

        self.pyramid_levels = cfg.detector.anchors.pyramid_levels
        self.strides        = cfg.detector.anchors.strides
        self.sizes          = cfg.detector.anchors.sizes
        self.ratios         = cfg.detector.anchors.ratios
        self.scales         = cfg.detector.anchors.scales
        self.shape = None
        self.P2 = None
        
        self.scale_step = 1 / (np.log2(self.scales[1]) - np.log2(self.scales[0]))
        
        # self.external_anchor_path = external_anchor_path
        # self.anchor_prior = anchor_prior
        # print(f"[anchors.py] external_anchor_path = {self.external_anchor_path}")
        # print(f"[anchors.py] self.anchor_prior = {anchor_prior}")

        self.anchors_mean_original = np.zeros([len(cfg.obj_types), len(self.scales), len(self.ratios), 6])
        self.anchors_std_original  = np.zeros([len(cfg.obj_types), len(self.scales), len(self.ratios), 6])
        self.anchors_mean_original = np.zeros([len(cfg.obj_types), len(self.scales) * len(self.pyramid_levels), len(self.ratios), 6])
        self.anchors_std_original  = np.zeros([len(cfg.obj_types), len(self.scales) * len(self.pyramid_levels), len(self.ratios), 6])
        
        if self.is_training_process: # For training process
            # Load mean and std file that calcualted during preprocessing
            self.anchors_avg_original = []
            self.anchors_std_original = []
            for i in range(len(cfg.obj_types)):
                npy_file = os.path.join(os.path.join(cfg.path.preprocessed_path, 'training', f'anchor_mean_{cfg.obj_types[i]}.npy'))
                std_file = os.path.join(os.path.join(cfg.path.preprocessed_path, 'training', f'anchor_std_{cfg.obj_types[i]}.npy'))
                # print(f"[anchor.py] Loading {npy_file}") # (16, 2, 6)
                # print(f"[anchor.py] Loading {std_file}") # (16, 2, 6)
                self.anchors_avg_original.append( np.load(npy_file) ) # [16, 2, 6] #[z,  sinalpha, cosalpha, w, h, l,]
                self.anchors_std_original.append( np.load(std_file) ) # [16, 2, 6] #[z,  sinalpha, cosalpha, w, h, l,]
                # self.anchors_avg_original[i] = np.load(npy_file) #[16, 2, 6] #[z,  sinalpha, cosalpha, w, h, l,]
                # self.anchors_std_original[i] = np.load(std_file) #[16, 2, 6] #[z,  sinalpha, cosalpha, w, h, l,]
            self.anchors_avg_original = np.array(self.anchors_avg_original) # (1, 32, 6)
            self.anchors_std_original = np.array(self.anchors_std_original) # (1, 32, 6)
            # print(f"self.anchors_avg_original = {self.anchors_avg_original.shape}") # (1, 32, 6)
            # print(f"self.anchors_std_original = {self.anchors_std_original.shape}") # (1, 32, 6)
        
        #######################
        ### External Anchor ###
        #######################, Get anchor from other sources
        if self.anchor_cfg.external_anchor_path != "":
            '''
            You need to pass in three files 
            2dbbox: (num_anchor, 4), [x1, y1, x2, y2]
            mean  : (num_anchor, 6), [z, sin2a, cos2a, w, h, l]
            std   : (num_anchor, 6), [z, sin2a, cos2a, w, h, l]
            '''
            anchor_fns = [os.path.join(self.anchor_cfg.external_anchor_path, fns) for fns in os.listdir(self.anchor_cfg.external_anchor_path)]
            print(f"Total {len(anchor_fns)} anchor files found in {self.anchor_cfg.external_anchor_path}")
            
            assert len(anchor_fns) == 1 or len(anchor_fns) == 3, "Too few or too much files, can only support one 2ddbox file or three files"
            
            # Load 2dbbox anchor file
            self.bbox2d_npy   = load_from_pkl_or_npy( next(f for f in anchor_fns if "2dbbox" in f) )
            
            # Load mean anchor file
            if self.anchor_cfg.anchor_prior:
                self.mean_npy = None
                self.std_npy  = None
            else:
                self.mean_npy = load_from_pkl_or_npy( next(f for f in anchor_fns if "mean" in f) )
                self.std_npy  = load_from_pkl_or_npy( next(f for f in anchor_fns if "std"  in f) )

    def anchors2indexes(self, anchors:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        """
            This will return the index in the 32 anchors
            computations in numpy: anchors[N, 4]
            return: sizes_int [N,]  ratio_ints [N, ]
        """
        # Find the the closert sizes
        if self.anchor_cfg.external_anchor_path == "":
            sizes = np.sqrt((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])) # area
            if self.cfg.detector.head.is_das:
                sizes_diff = sizes - (np.array(self.sizes[0]) * np.array(self.scales))[:, np.newaxis]
            else:
                sizes_diff = sizes - (np.array(self.sizes   ) * np.array(self.scales))[:, np.newaxis]
            sizes_int = np.argmin(np.abs(sizes_diff), axis=0)

            ratio =  (anchors[:, 3] - anchors[:, 1]) / (anchors[:, 2] - anchors[:, 0]) # aspect ratio
            ratio_diff = ratio - np.array(self.ratios)[:, np.newaxis]
            ratio_int = np.argmin(np.abs(ratio_diff), axis=0)

            return sizes_int, ratio_int

    def forward(self, image:torch.Tensor, calibs:List[np.ndarray]=[]): 
        
        shape = image.shape[2:] # torch.Size([288, 1280])
        if self.shape is None or not (shape == self.shape): # This block will only execute once during training
            self.shape  = image.shape[2:]
            image_shape = image.shape[2:]
            image_shape = np.array(image_shape)
            image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
            print(f"[anchors.py] image_shapes = {image_shapes}") # [array([18, 80]), .....]
            
            # compute anchors over all pyramid levels
            all_anchors = np.zeros((0, 4)).astype(np.float32)
            for idx, p in enumerate(self.pyramid_levels):
                
                ###########################
                ### Get anchors(2dbbox) ###
                ###########################
                if self.anchor_cfg.external_anchor_path == "": # Generate Anchor by yourself
                    anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
                
                else: # Use External Anchor
                    if type(self.bbox2d_npy) == dict: # Pyrimid anchors 
                        anchors = self.bbox2d_npy[p]
                    
                    else: # sinlge-level anchors
                        anchors = self.bbox2d_npy
                
                print(f"[anchors.py] anchors = {anchors.shape}") # (32, 4) From smallest to largest anchor
                print(f"[anchors.py] self.strides[idx] = {self.strides[idx]}")
                if self.cfg.detector.head.is_das:
                    if   p == 3: y_range = (6, 14) # (4, 14)
                    elif p == 4: y_range = (7,  9) # (7, 12)
                    elif p == 5: y_range = (4,  8) # (6, 9 )
                    shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors, y_range)
                else:
                    shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
                print(f"[anchors.py] shifted_anchors = {shifted_anchors.shape}") # (184320, 4), (46080, 4), (11520, 4)
                
                all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
            print(f"[anchors.py] all_anchors = {all_anchors.shape}") # (69210, 4) # (46080, 4)
            
            if self.is_training_process: # Only for training processs
                ############################################################
                ### Get mean and std of anchors from preprocessing files ###
                ############################################################
                if self.anchor_cfg.anchor_prior: # Use pre-processed mean and std file
                    print(f"Using pre-processed mean and std")
                    print(f"self.anchors_avg_original = {self.anchors_avg_original.shape}") # (1, 16, 2, 6)
                    if len(self.anchors_avg_original.shape) == 4: # The format is # (1, 16, 2, 6)
                        sizes_int, ratio_int = self.anchors2indexes(all_anchors) #　(46080,), (46080,)
                        self.anchor_means = image.new(self.anchors_avg_original[:, sizes_int, ratio_int]) #[types, N, 6], [1, 46080, 6]
                        self.anchor_stds  = image.new(self.anchors_std_original[:, sizes_int, ratio_int]) #[types, N, 6], [1, 46080, 6]
                    else: # The format is # (1, 32, 6)
                        self.anchor_means = image.new( np.expand_dims(np.tile(self.anchors_avg_original[0], (int(all_anchors.shape[0] / self.anchors_avg_original.shape[1]), 1)), axis=0) )
                        self.anchor_stds  = image.new( np.expand_dims(np.tile(self.anchors_std_original[0], (int(all_anchors.shape[0] / self.anchors_avg_original.shape[1]), 1)), axis=0) )
                    
                else: # Use External anchors file and use external mean and std files too
                    print(f"Using external anchor file ") 
                    if type(self.mean_npy) == dict: # Pyrimid anchor
                        anchor_means_list = []
                        anchor_stds_list  = []
                        for i_level, level in enumerate(self.pyramid_levels):
                            num_pixels = image_shapes[i_level][0] * image_shapes[i_level][1]
                            anchor_means_list.append( image.new( np.expand_dims(np.tile(self.mean_npy[level], (num_pixels, 1)), axis=0) ) )
                            anchor_stds_list. append( image.new( np.expand_dims(np.tile(self.std_npy[level],  (num_pixels, 1)), axis=0) ) )
                            # print(f"[anchor.py] anchor_means_list[-1].shape = {anchor_means_list[-1].shape}")
                        self.anchor_means = torch.cat(anchor_means_list, dim=1)
                        self.anchor_stds  = torch.cat(anchor_stds_list,  dim=1)
                        
                    else: # 
                        self.anchor_means = image.new( np.expand_dims(np.tile(self.mean_npy, (int(all_anchors.shape[0] / self.mean_npy.shape[0]), 1)), axis=0) )
                        self.anchor_stds  = image.new( np.expand_dims(np.tile(self.std_npy,  (int(all_anchors.shape[0] / self.mean_npy.shape[0]), 1)), axis=0) )
                
                print(f"[anchor.py] self.anchor_means = {self.anchor_means.shape}")
                print(f"[anchor.py] self.anchor_stds  = {self.anchor_stds.shape}")
                self.anchor_mean_std = torch.stack([self.anchor_means, self.anchor_stds], dim=-1).permute(1, 0, 2, 3) #[N, types, 6, 2]
            
            all_anchors = np.expand_dims(all_anchors, axis=0)
            if isinstance(image, torch.Tensor):
                self.anchors = image.new(all_anchors.astype(np.float32)) #[1, N, 4]
            elif isinstance(image, np.ndarray):
                self.anchors = torch.tensor(all_anchors.astype(np.float32)).cuda()
            self.anchors_image_x_center = self.anchors[0,:,0:4:2].mean(dim=1) #[N]
            self.anchors_image_y_center = self.anchors[0,:,1:4:2].mean(dim=1) #[N]
        
        #######################
        ### Get useful_mask ###
        #######################
        '''
        Use anchors' x3d, y3d to filter unreasonable anchors
        '''
        if calibs is not None and len(calibs) > 0:
            #P2 = calibs.P2 #[3, 4]
            #P2 = np.stack([calib for calib in calibs]) #[B, 3, 4]
            P2 = calibs #[B, 3, 4]
            
            # if P2 is the same than don't do anything
            if self.P2 is not None and torch.all(self.P2 == P2) and self.P2.shape == P2.shape:
                if self.is_training_process:
                    return self.anchors, self.useful_mask, self.anchor_mean_std
                else:
                    return self.anchors, self.useful_mask
            # print(f"Encounter different P2")
            self.P2 = P2
            N = self.anchors.shape[1]

            # Enable all anchors
            self.useful_mask = torch.ones([len(P2), N], dtype=torch.bool, device="cuda")
            
            # print(f"self.useful_mask = {self.useful_mask.shape}") # torch.Size([8, 61520])
            # print(f"[anchor.py] self.useful_mask = {torch.count_nonzero(self.useful_mask[0])}") # 9684
            
            if self.is_training_process:
                # Training 
                return self.anchors, self.useful_mask, self.anchor_mean_std
            else:
                # Preprocessing
                return self.anchors, self.useful_mask
        return self.anchors

    @property
    def num_anchors(self):
        # print(f"self.pyramid_levels = {self.pyramid_levels}") [4]
        # print(f"self.ratios = {self.ratios}") # [0.5 1. ]
        # print(f"self.scales = {self.scales}") # [2**0, 2**1/2, 2**1, .... 2**4]
        return len(self.pyramid_levels) * len(self.ratios) * len(self.scales)

    @property
    def num_anchor_per_scale(self):
        return len(self.ratios) * len(self.scales)

    @staticmethod
    def _deshift_anchors(anchors):
        """shift the anchors to zero base

        Args:
            anchors: [..., 4] [x1, y1, x2, y2]
        Returns:
            [..., 4] [x1, y1, x2, y2] as with (x1 + x2) == 0 and (y1 + y2) == 0
        """
        x1 = anchors[..., 0]
        y1 = anchors[..., 1]
        x2 = anchors[..., 2]
        y2 = anchors[..., 3]
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)

        return torch.stack([
            x1 - center_x,
            y1 - center_y,
            x2 - center_x,
            y2 - center_y
        ], dim=-1)

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales) # num_anchors = 32

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
       
    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales) # [32, 4]
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors

def shift(shape, stride, anchors, y_range = None):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    
    if y_range == None:
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    else:
        shift_y = (np.arange(y_range[0], y_range[1]) + 0.5) * stride
    # print(f"shift_x = {shift_x}")
    # [   8.   24.   40.   56.   72.   88.  104.  120.  136.  152.  168.  184.
    # 200.  216.  232.  248.  264.  280.  296.  312.  328.  344.  360.  376.
    # 392.  408.  424.  440.  456.  472.  488.  504.  520.  536.  552.  568.
    # 584.  600.  616.  632.  648.  664.  680.  696.  712.  728.  744.  760.
    # 776.  792.  808.  824.  840.  856.  872.  888.  904.  920.  936.  952.
    # 968.  984. 1000. 1016. 1032. 1048. 1064. 1080. 1096. 1112. 1128. 1144.
    # 1160. 1176. 1192. 1208. 1224. 1240. 1256. 1272.]
    
    # print(f"[anchor.py] shift_y = {shift_y}") # [8 24 40 56 72 88 104 120 136. 152. 168. 184. 200 216 232. 248. 264. 280.]

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()
    
    # print(f"shifts = {shifts}")
    # shifts = [[   8.    8.    8.    8.]
    #  [  24.    8.   24.    8.]
    #  [  40.    8.   40.    8.]
    #  ...
    #  [1240.  280. 1240.  280.]
    #  [1256.  280. 1256.  280.]
    #  [1272.  280. 1272.  280.]]
    #      x1    y1    x2    y2
    
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0] # 20
    K = shifts.shape[0] # 1440
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors
