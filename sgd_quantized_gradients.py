import os
import random
import shutil
import time
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist 
from torch import Tensor 
from torch.optim.optimizer import Optimizer, required 
from typing import List, Optional 
from quantization_supp.quant_modules import QuantLinear 
from quantization_supp.quant_utils import * 

def grad_buffer_update(model): 
    """  
    The function updates all the layer's grad buffer by the updated gradients across all layers in the model 
    The method is only called in single GPU training between loss.backward() and weights update 

    Parameter: 
    ---------- 
    model: the model that is training 

    Return: 
    ---------- 
    None 
    """ 
    
    with torch.no_grad(): 
        if model.emb_l is not None: 
            for emb_table in model.emb_l: 
                emb_table.embedding_grad_buffer.add_(emb_table.embedding_bag.weight.grad) 
        else: 
            raise Warning("Cannot find the list of embedding tables") 
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 

def grad_buffer_zeroing(model): 
    """ 
    The function zeros out all the grad buffer of the model's parameter 

    Parameter: 
    ---------- 
    model: the model that is training 

    Return: 
    ---------- 
    None 
    """ 

    if model.emb_l is not None: 
        for emb_table in model.emb_l: 
            emb_table.embedding_grad_buffer.zero_() 
    else: 
        raise Warning("Cannot find the list of embedding tables") 
    if model.bot_l is not None: 
        for layer_one in model.bot_l: 
            if isinstance(layer_one, QuantLinear): 
                layer_one.weight_grad_buffer.zero_() 
                layer_one.bias_grad_buffer.zero_() 
    else: 
        raise Warning("Cannot find the list of bottom linear layers") 
    if model.top_l is not None: 
        for layer_one in model.top_l: 
            if isinstance(layer_one, QuantLinear): 
                layer_one.weight_grad_buffer.zero_() 
                layer_one.bias_grad_buffer.zero_() 
    else: 
        raise Warning("Cannot find the list of top linear layers") 

def weights_update(model, lr): 
    """ 
    The function does step() function, and update all the parameters in the model 
    by using the buffer stored in each layer 

    Parameter: 
    ---------- 
    model: the model that is training 
    lr: the latest learning rate 
     
    Return: 
    ---------- 
    None 
    """ 
    with torch.no_grad(): 
        if model.emb_l is not None: 
            for emb_table in model.emb_l: 
                emb_table.embedding_bag.weight.data.add_(-lr * emb_table.embedding_grad_buffer) 
        else: 
            raise Warning("Cannot find the list of embedding tables") 
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear): 
                    layer_one.weight.data.add_(-lr * layer_one.weight_grad_buffer) 
                    layer_one.bias.data.add_(-lr * layer_one.bias_grad_buffer) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear): 
                    layer_one.weight.data.add_(-lr * layer_one.weight_grad_buffer) 
                    layer_one.bias.data.add_(-lr * layer_one.bias_grad_buffer) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 

def quantized_gradients_update(model, arg, lr, num_gpus): 
    """ 
    Communicating updates across all gpus, and do one step of weights update 

    Parameter 
    ---------- 
    model: the model that is training 
    arg: arguments from input settings 
    lr: the latest learning rate 

    Return 
    ---------- 
    None 
    """
    # no weight decay is used 
    with torch.no_grad(): 
        world_size = float(arg.world_size) 
        for name, param in model.named_parameters(): 
            update = param.grad # finding the gradient of the data by layer 
            dist.all_reduce(update, op = dist.ReduceOp.SUM) 
            update = update/num_gpus 
            '''
            print(name) 
            print(update.shape) 
            ''' 
            param.add_(update * (-lr[-1])) 

def clear_gradients(model): 
    """ 
    Clearing or zeroing out the gradients of all the parameters 
    of the model 

    Parameter 
    ---------- 
    model: the model that is training 

    Return 
    ---------- 
    None 
    """
    with torch.no_grad(): 
        for name, param in model.named_parameters(): 
            if param.grad is not None: 
                if param.grad.grad_fn is not None: 
                    param.grad.detach_() 
                else: 
                    param.grad.requires_grad_(False) 
                param.grad.zero_() 

def quantize_emb_grad(embedding_table, num_bits, parallel, num_gpus = None): 
    with torch.no_grad(): 
        if embedding_table.grad_fn is not None: 
            embedding_table.detach_() 
        else: 
            embedding_table.requires_grad_(False) 
        # finding scale 
        min = None 
        max = None 
        count = 0 
        used_rows_list = [] 
        for i in range(embedding_table.shape[0]): 
            if embedding_table[i][0] == 0: 
                if embedding_table[i][1] == 0: 
                    continue 
                    
            count += 1 
            used_rows_list.append(i) 

            if min is None: 
                min, _ = torch.min(embedding_table[i], dim = 0) 
            else: 
                new_min, _ = torch.min(embedding_table[i], dim = 0) 
                if new_min < min: 
                    min = new_min 
            if max is None: 
                max, _ = torch.max(embedding_table[i], dim = 0) 
            else: 
                new_max, _ = torch.max(embedding_table[i], dim = 0) 
                if new_max > max: 
                    max = new_max 
        print("sparsity level is {}".format(1 - float(count)/embedding_table.shape[0])) 
        n = 2 ** (num_bits - 1) - 1 

        scale = max(min.abs(), max.abs()) 
        scale = torch.clamp(scale, min = 1e-8)/n 

        if parallel: 
            dist.all_reduce(scale, dist.ReduceOp.SUM) 
            scale = scale/num_gpus 
        # quantize 
        return SymmetricQuantFunction.apply(embedding_table[used_rows_list], num_bits, scale), scale 

def quantize_linear_grad(weight, num_bits, parallel, num_gpus = None, per_channel = True): 
    with torch.no_grad(): 
        if weight.grad_fn is not None: 
            weight.detach_() 
        else: 
            weight.requires_grad_(False) 
        # finding scale 
        if per_channel: 
            w_min, _ = torch.min(weight, dim = 1, out = None) 
            w_max, _ = torch.max(weight, dim = 1, out = None) 
        else: 
            w_min = weight.min().expand(1) 
            w_max = weight.max().expand(1) 
        fc_scaling_factor = symmetric_linear_quantization_params(num_bits, w_min, w_max, per_channel) 
        if parallel: 
            dist.all_reduce(fc_scaling_factor, dist.ReduceOp.SUM) 
            fc_scaling_factor = fc_scaling_factor/num_gpus 
        # quantize 
        return SymmetricQuantFunction.apply(weight, num_bits, fc_scaling_factor), fc_scaling_factor 

def quantize_bias_grad(bias, num_bits, parallel, num_gpus = None): 
    with torch.no_grad(): 
        if bias.grad_fn is not None: 
            bias.detach_() 
        else: 
            bias.requires_grad_(False) 
        # finding scale 
        min = torch.min(bias, dim = 0, out = None) 
        max = torch.max(bias, dim = 0, out = None) 
        scale = symmetric_linear_quantization_params(num_bits, min, max) 
        if parallel: 
            dist.all_reduce(scale, dist.ReduceOp.SUM) 
            scale = scale/num_gpus 
        # quantize 
        return SymmetricQuantFunction.apply(bias, num_bits, scale), scale 
        
