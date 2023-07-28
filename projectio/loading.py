# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:58:18 2023

@author: Gavin
"""

import os, random
import torch

import numpy as np

from pconfig import COCO_DIR
from preprocessing import(
    gaussian_blur_batch, 
    gaussian_noise_batch, 
    split_with_padding
)
from torch.utils.data import Dataset, DataLoader, random_split
# from multiprocessing import Pool
from PIL import Image

def prepare_train_valid_loaders(
    loading_kwargs,
    dataloader_kwargs,
    dataset_kwargs
):
    
    train_set, valid_set = load_datasets(dataset_kwargs, **loading_kwargs, train=True)
    
    return prepare_dataloaders(train_set, valid_set, train=True, **dataloader_kwargs)



def prepare_test_loader(
    loading_kwargs,
    dataloader_kwargs,
    dataset_kwargs
):
    
    test_set = load_datasets(dataset_kwargs, **loading_kwargs, train=False)

    return prepare_dataloaders(test_set, train=False, **dataloader_kwargs)



def load_datasets(
        dataset_kwargs, 
        order_seed=0,
        dataset='COCO', 
        train=True, 
        train_frac=0.6, 
        valid_frac=0.2,
        load_limit=None
    ):
    
    random.seed(order_seed)
    np.random.seed(order_seed)
    torch.manual_seed(order_seed)
    
    if dataset.lower() == 'coco':
        data_dir = COCO_DIR
    else:
        raise ValueError(f'Argument dataset recieved invalid value "{dataset}"')
        
    all_img_fnames = [f'{data_dir}/{img_fname}' for img_fname in os.listdir(data_dir)]
    all_img_fnames = np.array(all_img_fnames)
    
    all_idxs = np.random.choice(len(all_img_fnames), size=len(all_img_fnames), replace=False)
    
    if load_limit is not None:
        all_idxs = all_idxs[:load_limit]
        all_img_fnames = all_img_fnames[all_idxs]
        
        all_idxs = np.arange(0, load_limit, 1)
        
        
    num_imgs = len(all_img_fnames)
        
    if train:
        img_fnames = all_img_fnames[all_idxs[:int(num_imgs * (train_frac + valid_frac))]]
        
        dataset = DirtyDataset(img_fnames, **dataset_kwargs)
        datasets = random_split(dataset, [
            train_frac / (train_frac + valid_frac), 
            valid_frac / (train_frac + valid_frac)
        ])
        
        return datasets
    else:
        img_fnames = all_img_fnames[all_idxs[int(num_imgs * (train_frac + valid_frac)):]]
        
        dataset = DirtyDataset(img_fnames, **dataset_kwargs)

        return dataset



def prepare_dataloaders(*datasets, train=True, **dataloader_kwargs):
    if train:
        loaders = [DataLoader(dataset, shuffle=True, **dataloader_kwargs) for dataset in datasets]
    
        return loaders
    else:
        loader = DataLoader(datasets[0], **dataloader_kwargs)
        
        return loader
        
        


class DirtyDataset(Dataset):
    
    def __init__(
        self,
        img_fnames,
        chunk_size=256,
        device='cpu',
        gaussian_kernel_min=3,
        gaussian_kernel_max=5,
        blur_sigma_min=1,
        blur_sigma_max=5,
        noise_sigma_min=0.01,
        noise_sigma_max=0.1
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        
        self.device = device
        
        self.gaussian_kernel_min = gaussian_kernel_min
        self.gaussian_kernel_max = gaussian_kernel_max
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_max = noise_sigma_max
        
        self.load_imgs(img_fnames)
            
        
        
    def __len__(self):
        return len(self.imgs)


    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        xs = self.imgs[idx]
        with torch.no_grad():
            ys = gaussian_blur_batch(
                xs.unsqueeze(dim=0),
                self.gaussian_kernel_min,
                self.gaussian_kernel_max,
                self.blur_sigma_min,
                self.blur_sigma_max
            )
            
            ys = gaussian_noise_batch(
                ys, 
                self.noise_sigma_min,
                self.noise_sigma_max
            )
            
            ys = ys.squeeze(dim=0)
            
        xs = xs.to(self.device).float()
        ys = ys.to(self.device).float()
        
        return xs, ys
    
    
    
    def load_img(self, img_fname):
        img = Image.open(img_fname)
        img = np.array(img) / 255
        
        img = np.rollaxis(img, 2)
        
        img_chunks = split_with_padding(img, self.chunk_size)
        img_chunks = torch.from_numpy(img_chunks)
        img_chunks = img_chunks.to(self.device)
        
        return img_chunks
    
    
    
    def load_imgs(self, img_fnames):
        self.imgs = [self.load_img(img_fname) for img_fname in img_fnames]
        self.imgs = torch.cat(self.imgs, dim=0)
            
        
                