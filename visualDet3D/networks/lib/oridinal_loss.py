# Reference:
# Good reference



# Bad Reference
# https://github.com/dontLoveBugs/SupervisedDepthPrediction/blob/master/dp/modules/losses/ordinal_regression_loss.py
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

class OrdinalRegressionLoss(torch.nn.Module):

    def __init__(self, ord_num, beta, discretization="SID"):
        super(OrdinalRegressionLoss, self).__init__()
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, depth):
        depth = torch.unsqueeze(depth, dim=1)
        
        N, _, H, W = depth.shape

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(depth.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(depth) / np.log(self.beta)
        else:
            label = self.ord_num * (depth - 1.0) / (self.beta - 1.0)
        
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(depth.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask < label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        
        return ord_c0, ord_c1

    def forward(self, prob, depth):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        N, C, H, W = prob.shape
        valid_mask = depth > 0.
        ord_c0, ord_c1 = self._create_ord_label(depth)
        # print(f"ord_c0 = {ord_c0.shape}")
        # print(f"ord_c1 = {ord_c1.shape}")
        
        logP   = torch.log(torch.clamp(    prob, min=1e-8))
        log1_P = torch.log(torch.clamp(1 - prob, min=1e-8))
        entropy = torch.sum(ord_c1*logP, dim=1) + torch.sum(ord_c0*log1_P, dim=1) # eq. (2)
        
        valid_mask = torch.squeeze(valid_mask, 1)
        loss = - entropy[valid_mask].mean()
        return loss

if __name__ == "__main__":
    ord_num = 5
    loss = OrdinalRegressionLoss(ord_num=ord_num, beta=80.0)
    gt = torch.rand(1, 1, 1) * 80
    gt[0, 0, 0] = 70
    print(f"gt = {gt}")
    
    prob = torch.rand(1, 2*ord_num, 1, 1)
    prob[0, 0, 0, 0] = 0
    prob[0, 1, 0, 0] = 0
    prob[0, 2, 0, 0] = 0
    prob[0, 3, 0, 0] = 0
    prob[0, 4, 0, 0] = 0
    
    prob[0, 5, 0, 0] = 1
    prob[0, 6, 0, 0] = 1
    prob[0, 7, 0, 0] = 1
    prob[0, 8, 0, 0] = 1
    prob[0, 9, 0, 0] = 1
    
    print(f"prob = {prob}")
    l = loss(prob, gt)
    print(l)
    
