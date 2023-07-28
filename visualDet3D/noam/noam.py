from visualDet3D.utils.util_kitti import AVG_LENTH, KITTI_Object
import math
import numpy as np

def noam_encode(loc_3d, P2):
    '''
    input = [[x3d, y3d, z3d, h, w, l], ...], P2
    ouput = [o1, o2, o3, ... o7] # Encoded result
    # TODO, use cx, cy to speed up
    '''
    # 
    objs = [KITTI_Object(f"Car NA NA NA NA NA NA NA {loc_3d[i_gt, 3]} {loc_3d[i_gt, 4]} {loc_3d[i_gt, 5]} {loc_3d[i_gt, 0]} {loc_3d[i_gt, 1]} {loc_3d[i_gt, 2]} 1000 NA", P2) for i_gt in range(loc_3d.shape[0])]
    
    # Draw annotation on image
    # for i_gt in range(loc_3d.shape[0]):\
    output = []
    for obj_i in objs:
        
        # 'category, truncated, occluded alpha, xmin, ymin, xmax, ymax, height, width, length, x3d, y3d, z3d, rot_y, score]
        obj_adj = {"farer": KITTI_Object(f"Inf NA NA NA NA NA NA NA {obj_i.h} {obj_i.w} {obj_i.l} {obj_i.x3d} {obj_i.y3d} {100} 1000 NA", P2), 
                   "close": KITTI_Object(f"Inf NA NA NA NA NA NA NA {obj_i.h} {obj_i.w} {obj_i.l} {obj_i.x3d} {obj_i.y3d}   {0} 1000 NA", P2), 
                   "right": KITTI_Object(f"Inf NA NA NA NA NA NA NA {obj_i.h} {obj_i.w} {obj_i.l} { 40} {obj_i.y3d} {obj_i.z3d} 1000 NA", P2), 
                   "lefft": KITTI_Object(f"Inf NA NA NA NA NA NA NA {obj_i.h} {obj_i.w} {obj_i.l} {-40} {obj_i.y3d} {obj_i.z3d} 1000 NA", P2)}

        # Find Nearby Object with strick critiron
        for obj_j in objs:
            # Ignore yourself
            if obj_i == obj_j: continue
            
            # Find far and close object       
            if abs(obj_i.x3d - obj_j.x3d) < AVG_LENTH/2:
                z3d_diff = obj_i.z3d - obj_j.z3d
                if   z3d_diff > 0 and abs(z3d_diff) < abs(obj_i.z3d - obj_adj["close"].z3d):
                    obj_adj["close"] = obj_j
                elif z3d_diff < 0 and abs(z3d_diff) < abs(obj_i.z3d - obj_adj["farer"].z3d):
                    obj_adj["farer"] = obj_j

            # Find right and left object   
            elif abs(obj_i.z3d - obj_j.z3d) < AVG_LENTH/2:
                x3d_diff = obj_i.x3d - obj_j.x3d
                if   x3d_diff < 0 and abs(x3d_diff) < abs(obj_i.x3d - obj_adj["right"].x3d):
                    obj_adj["right"] = obj_j
                elif x3d_diff > 0 and abs(x3d_diff) < abs(obj_i.x3d - obj_adj["lefft"].x3d):
                    obj_adj["lefft"]  = obj_j
        
        
        # Find Nearby Object with softer critiron
        for obj_j in objs:
            # Ignore yourself
            if obj_i == obj_j: continue
            
            # Find far and close object
            if abs(obj_i.x3d - obj_j.x3d) < AVG_LENTH:
                z3d_diff = obj_i.z3d - obj_j.z3d
                if   z3d_diff > 0 and abs(z3d_diff) < abs(obj_i.z3d - obj_adj["close"].z3d) and obj_adj["close"].category == 'Inf':
                    obj_adj["close"] = obj_j
                elif z3d_diff < 0 and abs(z3d_diff) < abs(obj_i.z3d - obj_adj["farer"].z3d) and obj_adj["farer"].category == 'Inf':
                    obj_adj["farer"]   = obj_j

            # Find right and left object   
            elif abs(obj_i.z3d - obj_j.z3d) < AVG_LENTH:
                x3d_diff = obj_i.x3d - obj_j.x3d
                if   x3d_diff < 0 and abs(x3d_diff) < abs(obj_i.x3d - obj_adj["right"].x3d) and obj_adj["right"].category == 'Inf':
                    obj_adj["right"] = obj_j
                elif x3d_diff > 0 and abs(x3d_diff) < abs(obj_i.x3d - obj_adj["lefft"].x3d) and obj_adj["lefft"].category == 'Inf':
                    obj_adj["lefft"]  = obj_j
        
        ###############################
        ### Output NOAM ground true ### Encode
        ###############################
        # TODO, Confident Score to indicate how sure the network is about the adjacency object
        # TODO, Using total_w - obj_i.cx to normalize might be a good idea
        # TODO, We can try to NOT use Inf to train network at all
        total_h = 288
        total_w = 1280
        
        # Farer Object
        farer_x = ( obj_adj['farer'].cx - obj_i.cx ) / total_w # normalize this to [0, 1]
        farer_y = ( obj_adj['farer'].cy - obj_i.cy ) / total_h # normalize this to [0, 1] where horizon is 1

        # Close Object
        if obj_adj['close'].category == "Inf":
            # Find clip point
            u1, v1 = obj_i.cx           , obj_i.cy
            u2, v2 = obj_adj['close'].cx, obj_adj['close'].cy
            a = (v2-v1)/(u2-u1) # slope, line formular: v = a*u + b 
            b = v1 - a*u1 # intercept
            # 
            ui1, vi1 = 0, b                      # i1 is left boundary of image
            ui2, vi2 = (total_h-b)/a , total_h   # i2 is bottom boundary of image
            ui3, vi3 = total_w, a*total_w + b    # i3 is right boudary of image
            #
            l1 = math.sqrt( (ui1 - u1)**2 + (vi1 - v1)**2 )
            l2 = math.sqrt( (ui2 - u1)**2 + (vi2 - v1)**2 )
            l3 = math.sqrt( (ui3 - u1)**2 + (vi3 - v1)**2 )
            #
            # Find the cloest boundary
            if   l1 == min(l1, l2, l3) and vi1 - v1 > 0: clip_p = (ui1, vi1)
            elif l2 == min(l1, l2, l3) and vi2 - v1 > 0: clip_p = (ui2, vi2)
            elif l3 == min(l1, l2, l3) and vi3 - v1 > 0: clip_p = (ui3, vi3)
            else: 
                # When the center of object is outside of image boundary, it might be able to find a positive solution 
                # The temporary solution is to allow the negatvie solution, since this issue is not very potent
                # print(f"[ERROR] Can't find solution (l1, l2, l3) = {(l1, l2, l3)}, obj_i.idx_img = {obj_i.idx_img}, obj_i.idx_line = {obj_i.idx_line}")
                if   l1 == min(l1, l2, l3): clip_p = (ui1, vi1)
                elif l2 == min(l1, l2, l3): clip_p = (ui2, vi2)
                elif l3 == min(l1, l2, l3): clip_p = (ui3, vi3)
            #
            close_x = ( clip_p[0] - obj_i.cx ) / total_w
            close_y = ( clip_p[1] - obj_i.cy ) / total_h
        else:
            close_x = ( obj_adj['close'].cx - obj_i.cx ) / total_w
            close_y = ( obj_adj['close'].cy - obj_i.cy ) / total_h
        
        # Right Object
        if obj_adj['right'].category == "Inf":
            right_x = (total_w - obj_i.cx) / total_w # Always positive
            right_y = 0.0
        else:
            right_x = (obj_adj['right'].cx - obj_i.cx) / total_w # Always positive
            right_y = (obj_adj['right'].cy - obj_i.cy) / total_h

        # Left Object
        if obj_adj['lefft'].category == "Inf":
            lefft_x = - obj_i.cx / total_w # # Always negative
            lefft_y = 0.0
        else:
            lefft_x = (obj_adj['lefft'].cx - obj_i.cx) / total_w # Always negative
            lefft_y = (obj_adj['lefft'].cy - obj_i.cy) / total_h

        obj_adj_target = [farer_x, farer_y,
                          close_x, close_y,
                          right_x, right_y, 
                          lefft_x, lefft_y]
        # print(obj_adj_target)
        output.append(obj_adj_target)
    return np.array(output)
            

        
def noam_decode(noam, cxcy, original_hw):
    output = np.zeros(8)
    # Decode
    for i_adj in range(8):
        if i_adj % 2 == 0: # even number
            output[i_adj] = noam[i_adj]*original_hw[1] + cxcy[0]
        else:
            output[i_adj] = noam[i_adj]*original_hw[0] + cxcy[1]
    return output
        