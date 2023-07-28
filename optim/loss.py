# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:20:18 2023

@author: Gavin
"""

import torch

from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

class PerceptualR50Loss:
    
    def __init__(self, device='cpu'):
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.eval()
        backbone.to(device)
        
        backbone = nn.Sequential(*list(backbone._modules.values())[:-2])
        
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone
        self.criterion = nn.MSELoss()



    def __call__(self, pred, true):
        if len(pred.shape) == 4 and pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            true = true.repeat(1, 3, 1, 1)
        elif len(pred.shape) == 3 and pred.shape[0] == 1:
            pred = pred.repeat(3, 1, 1)
            true = true.repeat(3, 1, 1)
        
        pred_fmap = self.backbone(pred)
        true_fmap = self.backbone(true)
        
        return self.criterion(pred_fmap, true_fmap)
    
    
    
class CompositeLoss:
    
    def __init__(
        self, 
        mse_weight=1, 
        perc_weight=1,
        device='cpu'
    ):
        self.mse_weight = mse_weight
        self.perc_weight = perc_weight
        
        self.mse = nn.MSELoss()
        self.perceptual = PerceptualR50Loss(device=device)
        
    
    
    def __call__(self, pred, true):
        mse = self.mse_weight * self.mse(pred, true)
        perceptual = self.perc_weight * self.perceptual(pred, true)
        
        return perceptual + mse