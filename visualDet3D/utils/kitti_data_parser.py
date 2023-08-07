import os
import numpy as np
from PIL import Image
from numba import jit
from numpy.linalg import inv
import cv2

def read_pc_from_bin(bin_path):
    """Load PointCloud data from bin file."""
    p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return p

def read_image(path):
    '''
    read image
    inputs:
        path(str): image path
    returns:
        img(np.array): [w,h,c]
    '''
    return np.array(Image.open(path, 'r'))

def read_depth(path:str)->np.ndarray:
    """ Read Ground Truth Depth Image
    
    Args:
        path: image path
    Return:
        depth image: floating image [H, W]
    """
    return (cv2.imread(path, -1)) / 256.0

@jit(nopython=True, cache=True)
def _leftcam2lidar(pts, Tr_velo_to_cam, R0_rect):
    '''
    transform the pts from the left camera frame to lidar frame
    pts_lidar  = Tr_velo_to_cam^{-1} @ R0_rect^{-1} @ pts_cam
    inputs:
        pts(np.array): [#pts, 3]
            points in the left camera frame
        Tr_velo_to_cam:[4, 4]
        R0_rect:[4, 4]
    '''

    hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
    pts_hT = np.ascontiguousarray(np.hstack((pts, hfiller)).T) #(4, #pts) 
    pts_lidar_T = np.ascontiguousarray(inv(Tr_velo_to_cam)) @ np.ascontiguousarray(inv(R0_rect)) @ pts_hT # (4, #pts)
    pts_lidar = np.ascontiguousarray(pts_lidar_T.T) # (#pts, 4)
    return pts_lidar[:, :3]

@jit(nopython=True, cache=True)
def _lidar2leftcam(pts, Tr_velo_to_cam, R0_rect):
    '''
    transform the pts from the lidar frame to the left camera frame
    pts_cam = R0_rect @ Tr_velo_to_cam @ pts_lidar
    inputs:
        pts(np.array): [#pts, 3]
            points in the lidar frame
    '''
    hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
    pts_hT = np.hstack((pts, hfiller)).T #(4, #pts)
    pts_cam_T = R0_rect @ Tr_velo_to_cam @ pts_hT # (4, #pts)
    pts_cam = pts_cam_T.T # (#pts, 4)
    return pts_cam[:, :3]

@jit(nopython=True, cache=True)
def _leftcam2imgplane(pts, P2):
    '''
    project the pts from the left camera frame to left camera plane
    pixels = P2 @ pts_cam
    inputs:
        pts(np.array): [#pts, 3]
        points in the left camera frame
    '''
    hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
    pts_hT = np.hstack((pts, hfiller)).T #(4, #pts)
    pixels_T = P2 @ pts_hT #(3, #pts)
    pixels = pixels_T.T
    pixels[:, 0] /= pixels[:, 2] + 1e-6
    pixels[:, 1] /= pixels[:, 2] + 1e-6
    return pixels[:, :2]

# spiderkiller changed index to name because index serve no purpose
def write_result_to_file(base_result_path:str, 
                         name:str, 
                         scores, 
                         bbox_2d, 
                         bbox_3d_state_3d=None, 
                         thetas=None, 
                         obj_types=['Car', 'Pedestrian', 'Cyclist'],
                         threshold=0.4):
    """Write Kitti prediction results of one frame to a file 

    Args:
        base_result_path (str): path to the result dictionary 
        # index (int): index of the target frame
        name (str): name of the image 
        scores (List[float]): A list or numpy array or cpu tensor of float for score
        bbox_2d (np.ndarray): numpy array of [N, 4]
        bbox_3d_state_3d (np.ndarray, optional): 3D stats [N, 7] [x_center, y_center, z_center, w, h, l, alpha]. Defaults to None.
        thetas (np.ndarray, optional): [N]. Defaults to None.
        obj_types (List[str], optional): List of string if object type names. Defaults to ['Car', 'Pedestrian', 'Cyclist'].
        threshold (float, optional): Threshold for selection samples. Defaults to 0.4.
    """
    # name = "%06d" % index
    text_to_write = ""
    file = open(os.path.join(base_result_path, name + '.txt'), 'w')
    if bbox_3d_state_3d is None:
        bbox_3d_state_3d = np.ones([bbox_2d.shape[0], 7], dtype=int)
        bbox_3d_state_3d[:, 3:6] = -1
        bbox_3d_state_3d[:, 0:3] = -1000
        bbox_3d_state_3d[:, 6]   = -10
    else:
        for i in range(len(bbox_2d)):
            bbox_3d_state_3d[i][1] = bbox_3d_state_3d[i][1] + 0.5*bbox_3d_state_3d[i][4] # kitti receive bottom center

    if thetas is None:
        thetas = np.ones(bbox_2d.shape[0]) * -10
    if len(scores) > 0:
        for i in range(len(bbox_2d)):
            if scores[i] < threshold:
                continue
            text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} \n').format(
                obj_types[i], bbox_3d_state_3d[i][-1], 
                bbox_2d[i][0], bbox_2d[i][1], bbox_2d[i][2], bbox_2d[i][3],
                bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][3], bbox_3d_state_3d[i][5],
                bbox_3d_state_3d[i][0], bbox_3d_state_3d[i][1], bbox_3d_state_3d[i][2],
                thetas[i], scores[i])

    file.write(text_to_write)
    file.close()

# KITTI
class KittiCalib:
    '''
    class storing KITTI calib data
        self.data(None/dict):keys: 'P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'
        self.R0_rect(np.array):  [4,4]
        self.Tr_velo_to_cam(np.array):  [4,4]
    '''
    def __init__(self, calib_path):
        self.path = calib_path
        self.data = None

    def read_calib_file(self):
        '''
        read KITTI calib file
        '''
        calib = dict()
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            calib[itm.split(':')[0]] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = [float(itm) for itm in v.split()]
        self.data = calib

        self.P2 = np.array(self.data['P2']).reshape(3,4)
        self.P3 = np.array(self.data['P3']).reshape(3,4)

        R0_rect = np.zeros([4, 4])
        R0_rect[0:3, 0:3] = np.array(self.data['R0_rect']).reshape(3, 3)
        R0_rect[3, 3] = 1
        self.R0_rect = R0_rect

        Tr_velo_to_cam = np.zeros([4, 4])
        Tr_velo_to_cam[0:3, :] = np.array(self.data['Tr_velo_to_cam']).reshape(3, 4)
        Tr_velo_to_cam[3, 3] = 1
        self.Tr_velo_to_cam = Tr_velo_to_cam

        return self
    
    def leftcam2lidar(self, pts):
        '''
        transform the pts from the left camera frame to lidar frame
        pts_lidar  = Tr_velo_to_cam^{-1} @ R0_rect^{-1} @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
                points in the left camera frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        return _leftcam2lidar(pts, self.Tr_velo_to_cam, self.R0_rect)

    def lidar2leftcam(self, pts):
        '''
        transform the pts from the lidar frame to the left camera frame
        pts_cam = R0_rect @ Tr_velo_to_cam @ pts_lidar
        inputs:
            pts(np.array): [#pts, 3]
                points in the lidar frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        return _lidar2leftcam(pts, self.Tr_velo_to_cam, self.R0_rect)

    def leftcam2imgplane(self, pts):
        '''
        project the pts from the left camera frame to left camera plane
        pixels = P2 @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
            points in the left camera frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        return _leftcam2imgplane(pts, self.P2)

class KittiLabel:
    '''
    class storing KITTI 3d object detection label
        self.data ([KittiObj])
    '''
    def __init__(self, label_path=None):
        self.path = label_path
        self.data = None

    def read_label_file(self, no_dontcare=True):
        '''
        read KITTI label file
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(KittiObj(s))
        if no_dontcare:
            self.data = list(filter(lambda obj: obj.type != "DontCare", self.data))
        return self

    def __str__(self):
        '''
        TODO: Unit TEST
        '''
        s = ''
        for obj in self.data:
            s += obj.__str__() + '\n'
        return s

    def equal(self, label, acc_cls, rtol):
        '''
        equal oprator for KittiLabel
        inputs:
            label: KittiLabel
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Notes: O(N^2)
        '''
        if len(self.data) != len(label.data):
            return False
        if len(self.data) == 0:
            return True
        bool_list = []
        for obj1 in self.data:
            bool_obj1 = False
            for obj2 in label.data:
                bool_obj1 = bool_obj1 or obj1.equal(obj2, acc_cls, rtol)
            bool_list.append(bool_obj1)
        return any(bool_list)

    def isempty(self):
        '''
        return True if self.data = None or self.data = []
        '''
        return self.data is None or len(self.data) == 0

class KittiObj():
    '''
    class storing a KITTI 3d object
    '''
    def __init__(self, s=None):
        self.type = None
        self.truncated = None
        self.occluded = None
        self.alpha = None
        self.bbox_l = None
        self.bbox_t = None
        self.bbox_r = None
        self.bbox_b = None
        self.h = None
        self.w = None
        self.l = None
        self.x = None
        self.y = None
        self.z = None
        self.ry = None
        self.score = None
        if s is None:
            return
        if len(s.split()) == 15: # data
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        elif len(s.split()) == 16: # result
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        else:
            raise NotImplementedError

    def __str__(self):
        if self.score is None:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.truncated, int(self.occluded), self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry)
        else:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.truncated, int(self.occluded), self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score)

class KittiData:
    '''
    class storing a frame of KITTI data
    '''
    def __init__(self, root_dir, idx, output_dict=None):
        '''
        inputs:
            root_dir(str): kitti dataset dir
            idx(str %6d): data index e.g. "000000"
            output_dict: decide what to output
        '''
        self.calib_path    = os.path.join(root_dir, "calib",       idx+'.txt')
        self.image2_path   = os.path.join(root_dir, "image_2",     idx+'.png')
        self.image3_path   = os.path.join(root_dir, 'image_3',     idx+'.png')
        self.label2_path   = os.path.join(root_dir, "label_2",     idx+'.txt')
        self.velodyne_path = os.path.join(root_dir, "velodyne",    idx+'.bin')
        self.depth_path    = os.path.join(root_dir, "image_depth", idx+'.png')
        
        self.output_dict = output_dict
        if self.output_dict is None: # Control what will be output when read_data()
            self.output_dict = {
                "calib": True,
                "image": True,
                "image_3": False,
                "label": True,
                "velodyne": True,
                "depth": False,
            }

    def read_data(self):
        '''
        read data
        returns:
            calib(KittiCalib)
            image(np.array): [w, h, 3]
            label(KittiLabel)
            pc(np.array): [# of points, 4]
                point cloud in lidar frame.
                [x, y, z]
                      ^x
                      |
                y<----.z
        '''
        calib = KittiCalib(self.calib_path).read_calib_file()     if self.output_dict["calib"]    else None
        image = read_image(self.image2_path)                      if self.output_dict["image"]    else None
        label = KittiLabel(self.label2_path).read_label_file()    if self.output_dict["label"]    else None
        pc    = read_pc_from_bin(self.velodyne_path)              if self.output_dict["velodyne"] else None
        depth = cv2.imread(self.depth_path, cv2.IMREAD_GRAYSCALE) if self.output_dict["depth"]    else None
        
        if 'image_3' in self.output_dict and self.output_dict['image_3']:
            image_3 = read_image(self.image3_path) if self.output_dict["image_3"] else None
            return calib, image, image_3, label, pc, depth
        else:
            return calib, image, label, pc, depth

if __name__ == "__main__":
    pts, Tr_velo_to_cam, R0_rect = np.zeros([10, 3]), np.eye(4), np.eye(4)

    points = _leftcam2lidar(pts, Tr_velo_to_cam, R0_rect)
    points = _lidar2leftcam(pts, Tr_velo_to_cam, R0_rect)

    P2 = np.zeros([3, 4])
    pixels = _leftcam2imgplane(pts, P2)

    print(points.shape)
