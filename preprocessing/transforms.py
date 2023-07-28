# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:03:42 2023

@author: Gavin
"""

import torch, kornia

import numpy as np
import torch.nn.functional as F

from numpy.lib.stride_tricks import sliding_window_view

def gaussian_blur_batch(
        imgs,
        kernel_size_min,
        kernel_size_max, 
        sigma_min,
        sigma_max
    ):
    blurred_imgs = kornia.filters.gaussian_blur2d(
        imgs,
        (kernel_size_min, kernel_size_max),
        (sigma_min, sigma_max)
    )
    
    return blurred_imgs



def split_with_padding(img, chunk_size):
    img_shape = np.array(img.shape)
    
    padded_img = np.zeros((3, *(np.ceil(img_shape[1:] / chunk_size) * chunk_size).astype(int)))
    padded_img[:, :img_shape[1], :img_shape[2]] = img
    
    def chunk_channel(img_channel):
        chunked_channel = sliding_window_view(img_channel, (chunk_size, chunk_size))
        chunked_channel = chunked_channel[::chunk_size, ::chunk_size]
        chunked_channel = chunked_channel.reshape(-1, chunk_size, chunk_size)
            
        return chunked_channel
    
    chunk_channel_vec = np.vectorize(chunk_channel, signature='(n,m)->(i,x,y)')
    
    chunked_img = chunk_channel_vec(padded_img)
    chunked_img = np.swapaxes(chunked_img, 0, 1)
    chunked_img = np.swapaxes(chunked_img, 2, 3)
    
    return chunked_img



def gaussian_noise_batch(imgs, sigma_min, sigma_max):
    sigma = np.random.rand() * (sigma_max - sigma_min) \
        + sigma_min
    
    return torch.clip_(
        imgs + torch.normal(mean=0, std=sigma, size=imgs.size()),
        min=0, max=1
    )