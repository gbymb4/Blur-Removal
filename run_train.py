# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:57:16 2023

@author: Gavin
"""

import os, sys, json, time, warnings
import torch, random

import numpy as np

from optim import DefaultOptimizer, compute_all_metrics
from projectio import (
    prepare_train_valid_loaders,
    prepare_test_loader,
    plot_and_save_metric,
    plot_and_save_visual,
    save_history_dict_and_model
)
from pconfig import (
    parse_config, 
    prepare_config,
    OUT_DIR
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def dump_metrics_plots(model, dataset, root_id, history):
    
    metrics_keys = list(history[0].keys())
    num_keys = len(metrics_keys)
    
    history_transpose = {key: [] for key in metrics_keys}
    for epoch_dict in history:
        for key, value in epoch_dict.items():
            history_transpose[key].append(value)

    for train_metric, valid_metric in zip(metrics_keys[:num_keys // 2], metrics_keys[num_keys // 2:]):
        train_vals = history_transpose[train_metric]
        valid_vals = history_transpose[valid_metric]
        
        metric = '_'.join(train_metric.split('_')[1:])
        
        print(f'plotting {metric} figure...')
        
        plot_fname = f'{OUT_DIR}/{dataset.lower()}/{type(model).__name__}/{root_id}/{metric}.pdf'
        plot_and_save_metric(train_vals, valid_vals, metric, plot_fname)
    


def dump_visualisations(
        model,
        dataset, 
        loader, 
        root_id,
        device,
        train=True, 
        plot_num=10
    ):
    num_saved = 0
    for batch in loader:
        xs, ys = batch
        
        xs = xs.to(device)
        ys = ys.to(device)
        
        preds = model(xs)
        
        for dirty, pred, true in zip(ys, preds, xs):
            if num_saved >= plot_num: return
            
            plot_root = f'{OUT_DIR}/{dataset.lower()}/{type(model).__name__}/{root_id}/'
            
            if not train: plot_root = f'{plot_root}/testing/visualisations'
            
            plot_fname = f'{plot_root}/visual{num_saved}.pdf'
            
            plot_and_save_visual(dirty, pred, true, plot_fname)
            num_saved += 1



def dump_test_items(model, dataset, test_loader, wavg_metrics, root_id, device):
    test_dir = f'{OUT_DIR}/{dataset.lower()}/{type(model).__name__}/{root_id}/testing'
    
    if not os.path.isdir(test_dir): os.mkdir(test_dir)
    
    metrics_fname = f'{test_dir}/test_performance.json'
    
    with open(metrics_fname, 'w') as file:
        json.dump(wavg_metrics, file)

    test_visuals_dir = f'{test_dir}/visualisations'

    if not os.path.isdir(test_visuals_dir): os.mkdir(test_visuals_dir)

    dump_visualisations(model, dataset, test_loader, root_id, device, train=False, plot_num=100)



def test_model(model, test_loader, device):
    test_num_imgs = 0
    
    model = model.eval()
    
    metrics_dict = {}
    
    for batch in test_loader:
        xs, ys = batch
        
        xs = xs.to(device)
        ys = ys.to(device)
        
        batch_size = len(batch)
    
        pred = model(xs)
        
        metric_scores = compute_all_metrics(pred, ys)
                
        for name, score in metric_scores.items():
            if name not in metrics_dict.keys():
                metrics_dict[name] = score * batch_size
            else:
                metrics_dict[name] += score * batch_size
        
        test_num_imgs += len(xs)
    
    wavg_metrics = {
        f'test_{name}': w_score / test_num_imgs for name, w_score in metrics_dict.items()
    }
    
    return wavg_metrics

        

def main():
    warnings.simplefilter('ignore')
    
    config_fname = sys.argv[1]
    config_dict = parse_config(config_fname)
    config_tup = prepare_config(config_dict)
    
    seed, dataset, model_type, device, train, test, *rest = config_tup
    model_kwargs, optim_kwargs, loading_kwargs, dataloader_kwargs, dataset_kwargs = rest
    
    set_seed(seed)
    
    root_id = int(time.time())
    
    if train:
        model = model_type(**model_kwargs).train().to(device)
        
        train_loader, valid_loader = prepare_train_valid_loaders(
            loading_kwargs,
            dataloader_kwargs,
            dataset_kwargs
        )
        
        optim = DefaultOptimizer(model, train_loader, valid_loader, device=device)
        history = optim.execute(**optim_kwargs)
        
        save_history_dict_and_model(dataset, model, root_id, config_dict, history)
        
        dump_metrics_plots(model, dataset, root_id, history)
        dump_visualisations(model, dataset, valid_loader, root_id, device)
    
    if test:
        test_loader = prepare_test_loader(
            loading_kwargs,
            dataloader_kwargs,
            dataset_kwargs
        )
        
        wavg_metrics = test_model(model, test_loader, device)
        
        dump_test_items(model, dataset, test_loader, wavg_metrics, root_id, device)
        
        
    
if __name__ == '__main__':
    main()