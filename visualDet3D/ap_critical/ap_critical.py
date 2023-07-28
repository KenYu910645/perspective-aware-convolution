from math import sin, cos, tan, sqrt, pi
import numpy as np

def sigmoid(x, offset=0, steep=1):
    return 1 / (1 + np.exp( (-x + offset) * steep) )

def cal_criticality(xzyaw, ap_mode="my"):
    '''
    xzyaw : np.array [[x3d, z3d, yaw], ......]
    '''
    # x3d, z3d, rot_y = xzyaw
    
    if ap_mode == "critical":
        # Reference: https://arxiv.org/abs/2203.02205
        V_MAX = 20 # m/s
        D_max, R_max, T_max = 20, 15, 8 # This parameter setting is suggest by the paper on section D
        x3d   = xzyaw[:, 0]
        z3d   = xzyaw[:, 1]
        rot_y = xzyaw[:, 2]
        
        d_ego_b = np.sqrt((x3d**2 + z3d**2))
        vx, vz  = np.cos(-rot_y)*V_MAX, np.sin(-rot_y)*V_MAX
        vz -= V_MAX # Relative speed against ego

        # Line formular : Ax + Bz + C = 0
        A = vz / vx
        B = -1
        C = z3d - A*x3d
        closetp_x = -A*C / (A**2 + B**2)
        closetp_z = -B*C / (A**2 + B**2)

        if (closetp_x - x3d) * (vx) > 0: # Same sign
            dt1 = (closetp_x - x3d) / vx
            dt2 = (closetp_z - z3d) / vz
            delta_t = dt1
            d_ego_c = np.abs(C) / np.sqrt(A**2 + B**2)
            # Don't know whether this will be a problem
            if np.abs(dt1 - dt2) > 1:
                print("[WARNING] dt1 dt2 is different, probably because x is too big")
        else: # ego and object are heading to differnet direction, so they won't collide in the future
            delta_t = float('inf')
            d_ego_c = float('inf')
        
        k_d = np.maximum(-(d_ego_b/D_max)**2 + 1, 0)
        k_r = np.maximum(-(d_ego_c/R_max)**2 + 1, 0)
        k_t = np.maximum(-(delta_t/T_max)**2 + 1, 0)
        kappa = 1 - (1-k_d) * (1-k_r) * (1-k_t)
        # print(f"(k_d, k_r, k_t, kappa) = {(k_d, k_r, k_t, self.kappa)}")
    
    elif ap_mode == "my":
        ##############
        ### AP_SCT ###
        ##############
        # theta = -rot_y # KITTI's angle is close-wise postive
        # p_collision = (0, z3d - x3d*tan(theta))
        # d_ego_b = sqrt((x3d**2 + z3d**2))
        # d_ego_c = abs( z3d - x3d*tan(theta) )
        
        d_ego_b = np.sqrt((xzyaw[:, 0]**2 + xzyaw[:, 1]**2)) # sqrt(x3d**2 + y3d**2)
        d_ego_c = np.abs ( xzyaw[:, 1] - xzyaw[:, 0]*np.tan(-xzyaw[:, 2]) )
        D_max, R_max = 80, 80
        
        # k_d = max(0, -(d_ego_b/D_max)**2 + 1)
        # k_r = max(0, -(d_ego_c/R_max)**2 + 1)
        k_d = 1 - sigmoid(d_ego_b, offset=40, steep=0.1)
        k_r = 1 - sigmoid(d_ego_c, offset=40, steep=0.1)
        
        kappa = 1 - (1-k_d) * (1-k_r)
        
        # print(f"(k_d, k_r, kappa) = {(k_d, k_r, kappa)}")
    # return kappa, p_collision
    return kappa