# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:58:23 2023

@author: Gavin
"""

import os, json, yaml
import torch

import matplotlib.pyplot as plt

from torch import nn
from pconfig import OUT_DIR
from matplotlib.ticker import AutoMinorLocator

def plot_and_save_metric(train, valid, metric, fname):
    fig, ax = plt.subplots(figsize=(8, 6))

    epochs = list(range(1, len(train) + 1))

    ax.plot(epochs, train, label='Train', alpha=0.7)
    ax.plot(epochs, valid, label='Validation', alpha=0.7)   
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='major', length=4)
    ax.tick_params(which='minor', length=2, color='r')
    
    ax.legend()
    ax.grid(axis='y', c='white')
    
    ax.set_facecolor('whitesmoke')
    
    metric_name = (' '.join(metric.split('_'))).title()
    
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18)
    
    plt.savefig(fname)
    plt.show()
    
    

def plot_and_save_visual(img, pred, true, fname):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    
    for ax, tensor, title in zip(axs, [img, pred, true], ['Dirty', 'Prediction', 'GT']):
        ax.axis('off')
        ax.imshow(tensor.detach().cpu().swapaxes(0, 2))
        ax.set_title(title, fontsize=18)
    
    plt.savefig(fname)
    plt.show()
    
    
    
def save_history_dict_and_model(
    dataset: str,
    model: nn.Module,
    root_id: int,
    config: dict,
    history: dict
) -> None:
    dataset = dataset.lower()
    model_name = type(model).__name__

    if not os.path.isdir(OUT_DIR): os.mkdir(OUT_DIR)

    save_root_dir = f'{OUT_DIR}/{dataset}'
    if not os.path.isdir(save_root_dir): os.mkdir(save_root_dir)
    
    save_model_dir = f'{save_root_dir}/{model_name}'
    if not os.path.isdir(save_model_dir): os.mkdir(save_model_dir)

    save_dir = f'{save_model_dir}/{root_id}'
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    with open(f'{save_dir}/history.json', 'w') as file:
        json.dump(history, file)

    with open(f'{save_dir}/config.yaml', 'w') as file:
        yaml.safe_dump(config, file)
        
    for param in model.parameters():
        param.requires_grad = True

    torch.save(model.state_dict(), f'{save_dir}/model')