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

def grad_buffer_update(model, number_of_gpus): 
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
                emb_table.embedding_grad_buffer.add_(emb_table.embedding_bag.weight.grad/number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of embedding tables") 
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad/number_of_gpus) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad/number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad/number_of_gpus) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad/number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 

def grad_buffer_update_added_quantization(model, number_of_gpus): 
    """ 
    The function updates all the layer's grad buffer by the updated gradients across all layers in the model 
    The updates are quantized. 
    The method is only called in single GPU training between loss.backward() and weights update 

    Parameter: 
    ---------- 
    model: the model that is training 

    Return: 
    ---------- 
    None 
    """ 
    # dequantize 
    # do conversion to floating point full precision number back in weight_update function 
    with torch.no_grad(): 
        if model.emb_l is not None: 
            for emb_table in model.emb_l: 
                # quantize 
                if not torch.is_nonzero(emb_table.emb_scaling_factor): # check if scale is set to zero 
                    buffer_changes, scale = quantize_emb_grad(emb_table.embedding_bag.weight.grad, num_bits = 16, parallel = False) 
                    emb_table.emb_scaling_factor = scale 
                else: 
                    buffer_changes, _ = quantize_emb_grad(emb_table.embedding_bag.weight.grad, num_bits = 16, parallel = False, scale = emb_table.emb_scaling_factor) 
                emb_table.embedding_grad_buffer.add_(buffer_changes) # buffer accumulates integer tensors, scales handles the batch size 
        else: 
            raise Warning("Cannot find the list of embedding tables") 
        
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear): 
                    # weights 
                    if not torch.is_nonzero(torch.sum(layer_one.weight_scaling_factor, dim = 0)):  # check if scale is set to zero 
                        buffer_changes, scale = quantize_linear_grad(layer_one.weight.grad, num_bits = 16, parallel = False) 
                        layer_one.weight_scaling_factor = scale 
                    else: 
                        buffer_changes, _ = quantize_linear_grad(layer_one.weight.grad, num_bits = 16, parallel = False, scale = layer_one.weight_scaling_factor) 
                    layer_one.weight_grad_buffer.add_(buffer_changes) 

                    # bias 
                    if not torch.is_nonzero(torch.sum(layer_one.bias_scaling_factor, dim = 0)):  # check if scale is set to zero 
                        buffer_changes, scale = quantize_bias_grad(layer_one.bias.grad, num_bits = 16, parallel = False) 
                        layer_one.bias_scaling_factor = scale 
                    else: 
                        buffer_changes, _ = quantize_bias_grad(layer_one.bias.grad, num_bits = 16, parallel = False, scale = layer_one.bias_scaling_factor) 
                    layer_one.bias_grad_buffer.add_(buffer_changes) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear): 
                    # weights 
                    if not torch.is_nonzero(torch.sum(layer_one.weight_scaling_factor, dim = 0)): # check if scale is set to zero 
                        buffer_changes, scale = quantize_linear_grad(layer_one.weight.grad, num_bits = 16, parallel = False) 
                        layer_one.weight_scaling_factor = scale 
                    else: 
                        buffer_changes, _ = quantize_linear_grad(layer_one.weight.grad, num_bits = 16, parallel = False, scale = layer_one.weight_scaling_factor) 
                    layer_one.weight_grad_buffer.add_(buffer_changes) 

                    # bias 
                    if not torch.is_nonzero(torch.sum(layer_one.bias_scaling_factor, dim = 0)): # check if scale is set to zero 
                        buffer_changes, scale = quantize_bias_grad(layer_one.bias.grad, num_bits = 16, parallel = False) 
                        layer_one.bias_scaling_factor = scale 
                    else: 
                        buffer_changes, _ = quantize_bias_grad(layer_one.bias.grad, num_bits = 16, parallel = False, scale = layer_one.bias_scaling_factor) 
                    layer_one.bias_grad_buffer.add_(buffer_changes) 
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
            emb_table.emb_scaling_factor.zero_() # zero out the scale 
    else: 
        raise Warning("Cannot find the list of embedding tables") 
    if model.bot_l is not None: 
        for layer_one in model.bot_l: 
            if isinstance(layer_one, QuantLinear): 
                # weights 
                layer_one.weight_grad_buffer.zero_() 
                layer_one.weight_scaling_factor.zero_() 

                # bias 
                layer_one.bias_grad_buffer.zero_() 
                layer_one.bias_scaling_factor.zero_() 
    else: 
        raise Warning("Cannot find the list of bottom linear layers") 
    if model.top_l is not None: 
        for layer_one in model.top_l: 
            if isinstance(layer_one, QuantLinear): 
                # weights 
                layer_one.weight_grad_buffer.zero_() 
                layer_one.weight_scaling_factor.zero_() 

                # bias 
                layer_one.bias_grad_buffer.zero_() 
                layer_one.bias_scaling_factor.zero_() 
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

def weights_update_added_quantization(model, lr, num_gpus): 
    """ 
    The function does step() function, and update all all the parameters in the model 
    by using the buffer stored in each layer 
    The function uses buffer that contains integer, which are then dequantized and used 
    to update the weights 

    Parameter: 
    ---------- 
    model: the model that is training 
    lr: the latest learning rate 
    num_gpus: the number of gpus 

    Return: 
    ---------- 
    None 
    """ 
    with torch.no_grad(): 
        if model.emb_l is not None: 
            for emb_table in model.emb_l: 
                print(emb_table.embedding_grad_buffer.coalesce().values()[: 20]) 
                weight_update = emb_table.embedding_grad_buffer * (emb_table.emb_scaling_factor/num_gpus) # dequantize 
                emb_table.embedding_bag.weight.data.add_(-lr * weight_update) # update 
        else: 
            raise Warning("Cannot find the list of embedding tables") 
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear): 
                    # weight 
                    weight_update = layer_one.weight_grad_buffer * (layer_one.weight_scaling_factor.view(-1, 1)/num_gpus) # dequantize 
                    layer_one.weight.data.add_(-lr * weight_update) # update 

                    # bias 
                    bias_update = layer_one.bias_grad_buffer * (layer_one.bias_scaling_factor/num_gpus) # dequantize 
                    layer_one.bias.data.add_(-lr * bias_update) # update 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear): 
                    # weight 
                    weight_update = layer_one.weight_grad_buffer * (layer_one.weight_scaling_factor.view(-1, 1)/num_gpus) # dequantize 
                    layer_one.weight.data.add_(-lr * weight_update) # update 

                    # bias 
                    bias_update = layer_one.bias_grad_buffer * (layer_one.bias_scaling_factor/num_gpus) # dequantize 
                    layer_one.bias.data.add_(-lr * bias_update) # update 
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

def quantize_emb_grad(embedding_table, num_bits, parallel, num_gpus = None, scale = None): 
    with torch.no_grad(): 
        if embedding_table.grad_fn is not None: 
            embedding_table.detach_() 
        else: 
            embedding_table.requires_grad_(False) 
        # finding scale 
        print(embedding_table.coalesce().indices().shape) 
        embedding_table = embedding_table.coalesce() 
        if scale is None: 
            scale = symmetric_linear_quantization_param_two(num_bits, embedding_table.values(), None, None, None) 
            print(scale) 

        if parallel: 
            dist.all_reduce(scale, dist.ReduceOp.SUM) 
            scale = scale/num_gpus 
        scale = scale.view(-1) 
        # quantize 
        return torch.sparse_coo_tensor(embedding_table.indices(), SymmetricQuantFunction.apply(embedding_table.values(), num_bits, scale), size = embedding_table.size(), device = embedding_table.device), scale 

def quantize_linear_grad(weight, num_bits, parallel, num_gpus = None, per_channel = True, scale = None): 
    with torch.no_grad(): 
        if weight.grad_fn is not None: 
            weight.detach_() 
        else: 
            weight.requires_grad_(False) 
        if scale is None: 
            # finding scale 
            if per_channel: 
                w_min, _ = torch.min(weight, dim = 1, out = None) 
                w_max, _ = torch.max(weight, dim = 1, out = None) 
            else: 
                w_min = weight.min().expand(1) 
                w_max = weight.max().expand(1) 
            fc_scaling_factor = symmetric_linear_quantization_params(num_bits, w_min, w_max, per_channel) 
            print("linear weight scale shape: {}".format(fc_scaling_factor.shape)) 
        else: 
            fc_scaling_factor = scale 
        if parallel: 
            dist.all_reduce(fc_scaling_factor, dist.ReduceOp.SUM) 
            fc_scaling_factor = fc_scaling_factor/num_gpus 
        # quantize 
        return SymmetricQuantFunction.apply(weight, num_bits, fc_scaling_factor), fc_scaling_factor 

def quantize_bias_grad(bias, num_bits, parallel, num_gpus = None, scale = None): 
    with torch.no_grad(): 
        if bias.grad_fn is not None: 
            bias.detach_() 
        else: 
            bias.requires_grad_(False) 
        if scale is None: 
            # finding scale 
            bias_min, _ = torch.min(bias, dim = 0, out = None) 
            bias_max, _ = torch.max(bias, dim = 0, out = None) 
            scale = symmetric_linear_quantization_params(num_bits, bias_min, bias_max) 
        if parallel: 
            dist.all_reduce(scale, dist.ReduceOp.SUM) 
            scale = scale/num_gpus 
        # quantize 
        return SymmetricQuantFunction.apply(bias, num_bits, scale), scale 
        
