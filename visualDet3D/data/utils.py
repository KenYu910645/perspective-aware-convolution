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
                         noam=None,
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
            if noam == None:
                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} \n').format(
                    obj_types[i], bbox_3d_state_3d[i][-1], 
                    bbox_2d[i][0], bbox_2d[i][1], bbox_2d[i][2], bbox_2d[i][3],
                    bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][3], bbox_3d_state_3d[i][5],
                    bbox_3d_state_3d[i][0], bbox_3d_state_3d[i][1], bbox_3d_state_3d[i][2],
                    thetas[i], scores[i])
            else:
                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n').format(
                    obj_types[i], bbox_3d_state_3d[i][-1], 
                    bbox_2d[i][0], bbox_2d[i][1], bbox_2d[i][2], bbox_2d[i][3],
                    bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][3], bbox_3d_state_3d[i][5],
                    bbox_3d_state_3d[i][0], bbox_3d_state_3d[i][1], bbox_3d_state_3d[i][2],
                    thetas[i], scores[i],
                    noam[i][0], noam[i][1], noam[i][2], noam[i][3], 
                    noam[i][4], noam[i][5], noam[i][6], noam[i][7])
    file.write(text_to_write)
    file.close()

if __name__ == "__main__":
    pts, Tr_velo_to_cam, R0_rect = np.zeros([10, 3]), np.eye(4), np.eye(4)

    points = _leftcam2lidar(pts, Tr_velo_to_cam, R0_rect)
    points = _lidar2leftcam(pts, Tr_velo_to_cam, R0_rect)

    P2 = np.zeros([3, 4])
    pixels = _leftcam2imgplane(pts, P2)

    print(points.shape)
