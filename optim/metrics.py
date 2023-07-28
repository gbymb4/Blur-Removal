# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:18:01 2023

@author: Gavin
"""

import torch

import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure

def psnr(pred, true):
    mse = F.mse_loss(pred, true)
    psnr_val = 20 * torch.log10(torch.max(true) / torch.sqrt(mse))
    
    return psnr_val.item()



def ssim(pred, true):
    ssim = StructuralSimilarityIndexMeasure(data_range=1).to(pred.device)
    ssim_val = 1 - ssim(pred, true)
    
    return ssim_val.item()
    


def compute_all_metrics(pred, true):
    results = {}
    
    results['psnr'] = psnr(pred, true)
    results['ssim'] = ssim(pred, true)

    return results