# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:30:37 2023

@author: Gavin
"""

import yaml

from copy import deepcopy
from models.unet import UNet

def parse_config(fname: str) -> dict:
    with open(fname, 'r') as file:
        config = yaml.safe_load(file)

    return config



def prepare_config(
    config: dict
) -> tuple:
    seed = config['seed']
    dataset = deepcopy(config['dataset'])
    model_name = deepcopy(config['model'])

    if model_name.lower() == 'unet':
        model = UNet
    else:
        raise ValueError(f'Invalid model type "{model_name}" in config file.')

    device = deepcopy(config['device'])
    train = deepcopy(config['train'])
    test = deepcopy(config['test'])
    
    model_kwargs = deepcopy(config['model_arguments'])
    optim_kwargs = deepcopy(config['optimizer_arguments'])
    loading_kwargs = deepcopy(config['loading_arguments'])
    dataloader_kwargs = deepcopy(config['dataloader_arguments'])
    dataset_kwargs = deepcopy(config['dataset_arguments'])

    args = (seed, dataset, model, device, train, test)
    kwargs = (model_kwargs, optim_kwargs, loading_kwargs, dataloader_kwargs, dataset_kwargs)

    out = (*args, *kwargs)

    return out