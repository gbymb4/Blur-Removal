# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:59:38 2023

@author: Gavin
"""

import time
import torch

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from typing import List
from .loss import CompositeLoss
from .metrics import compute_all_metrics

class DefaultOptimizer:
    
    def __init__(self,
        model: nn.Module, 
        train: DataLoader, 
        valid: DataLoader,
        device: str='cpu'
    ) -> None:
        super().__init__()
        
        self.model = model
        self.train_loader = train
        self.valid_loader = valid
        self.device = device

        
        
    def execute(self,
        epochs=100,
        lr=1e-5,
        valid_freq=10, 
        mse_weight=1,
        perc_weight=1,
        verbose=True
    ) -> List[dict]:
        history = []

        start = time.time()

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = CompositeLoss(
            mse_weight=mse_weight,
            perc_weight=perc_weight,
            device=self.device
        )
        
        print('#'*32)
        print('beginning BP training loop...')
        print('#'*32)
        
        for i in range(epochs):
            epoch = i + 1
            
            if verbose and i % 10 == 0:
                print(f'executing epoch {epoch}...', end='')
                
            history_record = {}
            
            train_num_imgs = 0
            train_loss = 0
            
            model = self.model.train()
            
            metrics_dict = {}

            for batch in self.train_loader:
                xs, ys = batch
                
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                
                batch_size = len(batch)

                optim.zero_grad()

                pred = model(xs)
                
                loss = criterion(pred, ys)
                loss.backward()
                train_loss = loss.item()

                metric_scores = compute_all_metrics(pred, ys)
                        
                for name, score in metric_scores.items():
                    if name not in metrics_dict.keys():
                        metrics_dict[name] = score * batch_size
                    else:
                        metrics_dict[name] += score * batch_size

                train_num_imgs += len(xs)

                optim.step()
            
            history_record['train_loss'] = train_loss
            history_record['train_norm_loss'] = train_loss / train_num_imgs
            
            wavg_metrics = {
                f'train_{name}': w_score / train_num_imgs for name, w_score in metrics_dict.items()
            }
            
            history_record.update(wavg_metrics)

            if i % valid_freq == 0 or epoch == epochs:
                valid_num_slides = 0
                valid_loss = 0
    
                model = self.model.eval()
                
                metrics_dict = {}
    
                for batch in self.valid_loader:
                    xs, ys = batch
                    
                    xs = xs.to(self.device)
                    ys = ys.to(self.device)
                    
                    batch_size = len(batch)

                    optim.zero_grad()

                    pred = model(xs)
                    
                    loss = criterion(pred, ys)
                    valid_loss = loss.item()

                    metric_scores = compute_all_metrics(pred, ys)
                            
                    for name, score in metric_scores.items():
                        if name not in metrics_dict.keys():
                            metrics_dict[name] = score * batch_size
                        else:
                            metrics_dict[name] += score * batch_size

                    valid_num_slides += len(xs)

                history_record['valid_loss'] = valid_loss
                history_record['valid_norm_loss'] = valid_loss / valid_num_slides
                
                wavg_metrics = {
                    f'valid_{name}': w_score / valid_num_slides for name, w_score in metrics_dict.items()
                }
                
                history_record.update(wavg_metrics)

            history.append(history_record)

            if verbose and i % 10 == 0 and epoch != epochs:
                print('done')
                print(f'epoch {epoch} training statistics:')
                print('\n'.join([f'->{key} = {value:.4f}' for key, value in history_record.items()]))
                print('-'*32)
            
        print('#'*32)
        print('finished BP training loop!')
        print('final training statistics:')
        print('\n'.join([f'->{key} = {value:.4f}' for key, value in history[-1].items()]))
        print('#'*32)

        end = time.time()

        print(f'total elapsed time: {end-start}s')

        return history