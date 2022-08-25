import argparse
import os
import random
import shutil
import time
import logging
import warnings

import dlrm_data_pytorch as dp 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models 

def main(args): 
    parser = argparse.ArgumentParser( 
        description = "Train a simple network" 
    ) 
    parser.add_argument("-n", "--nodes", default = 1, 
                        type = int, metavar = "N") 
    parser.add_argument("-g", "--gpus", default = 1, type = int, 
                        help = "number of gpus per node") 
    parser.add_argument("-nr", "--nr", default = 0, type = int, 
                        help = "ranking within the nodes") 

    args = parser.parse_args() 
    args.world_size = args.gpus * args.nodes 
    os.environ['MASTER_ADDR'] = '169.229.49.63' 
    os.environ['MASTER_PORT'] = '29571' 
    mp.spawn(train, nprocs = args.gpus, args = (args,)) 

def train(gpu, args): 
    rank = args.nr * args.gpus + gpu 
    dist.init_process_grpup(
        backend = "gloo", 
        init_method = 'env://', 
        world_size = args.world_size, 
        rank = rank 
    ) 

    torch.manual_seed(0) 
    torch.cuda.set_device(gpu) 
    batch_size = args.mini_batch_size 
    torch.set_printoptions(profile = "full") 

    use_gpu = args.use_gpu and torch.cuda.is_available() 
    '''
    print("use gpu" if use_gpu else "not use gpu") 
    ''' 
    
    if use_gpu: 
        ngpus = 1 
        device = torch.device("cuda", gpu) 
        if gpu != args.local_rank: 
            print("Warning: local_rank gpu mismatch") 
        print("{} out of {} (GPU)".format(torch.cuda.device_count(), args.local_rank)) # TODO think about using cpu and change code 
    else: 
        device = torch.device("cpu") 
        print("Using CPU...") 
    '''
    print(device) 
    ''' 
    
    ### prepare training data ### 
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype = int, sep = "-") 
    
    train_dataset, test_dataset = dp.make_criteo_data_and_loaders_two(args) 
    
    # train sampler 
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = args.world_size, rank = rank) 
    '''
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas = args.world_size, rank = rank) 
    ''' 
    
    collate_wrapper_criteo = dp.collate_wrapper_criteo_offset 
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        batch_size = batch_size, 
        shuffle = False, 
        num_workers = 0, 
        pin_memory = True, 
        sampler = train_sampler, 
        collate_fn = collate_wrapper_criteo, 
        drop_last = False 
    ) 
    '''
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        batch_size = args.test_mini_batch_size, 
        shuffle = False, 
        num_workers = args.test_num_workers, 
        pin_memory = True, 
        sampler = test_sampler, 
        collate_fn = collate_wrapper_criteo, 
        drop_last = False 
    )
    ''' 
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        batch_size = args.test_mini_batch_size, 
        shuffle = False, 
        num_workers = args.test_num_workers, 
        pin_memory = True, 
        collate_fn = collate_wrapper_criteo, 
        drop_last = False 
    ) 
    
    nbatches = args.num_batches if args.num_batches > 0 else len(train_loader) 
    '''
    if (gpu == 0): 
        print("number of batches is ", nbatches) 
    ''' 
    nbatches_test = len(test_loader) 
    
    ln_emb = train_dataset.counts 
    if args.max_ind_range > 0:
        ln_emb = np.array(
            list(
                map(
                    lambda x: x if x < args.max_ind_range else args.max_ind_range,
                    ln_emb,
                )
            )
        )
    else:
        ln_emb = np.array(ln_emb) 
    m_den = train_dataset.m_den 
    ln_bot[0] = m_den 
    
    args.ln_emb = ln_emb.tolist() 
