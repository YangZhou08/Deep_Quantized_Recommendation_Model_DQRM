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
from quantization_supp.full_precision_modules import LinearCompressedGrad 
from quantization_supp.quant_modules_not_quantize_grad import QuantLinear 
from quantization_supp.quant_utils import * 

total_comm_time = [0 for i in range(26)] 

def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time() 

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
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad/number_of_gpus) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad/number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad/number_of_gpus) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad/number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 

def grad_buffer_update_added_quantization(model, number_of_gpus, emb_grad_quantized = True): 
    """ 
    The function updates all the layer's grad buffer by the updated gradients across all layers in the model 
    The updates are quantized. 
    The method is only called in single GPU training between loss.backward() and weights update 

    Parameter: 
    ---------- 
    model: the model that is training 
    number_of_gpus: the number of gpus that are used or simulatedly used 

    Return: 
    ---------- 
    None 
    """ 
    # dequantize 
    # do conversion to floating point full precision number back in weight_update function 
    with torch.no_grad(): 
        if emb_grad_quantized: 
            if model.emb_l is not None: 
                for emb_table in model.emb_l: 
                    # quantize 
                    if not torch.is_nonzero(emb_table.emb_scaling_factor): # check if scale is set to zero 
                        buffer_changes, scale = quantize_emb_grad(emb_table.embedding_bag.weight.grad, num_bits = 8, parallel = False) 
                        emb_table.emb_scaling_factor = scale 
                    else: 
                        buffer_changes, _ = quantize_emb_grad(emb_table.embedding_bag.weight.grad, num_bits = 8, parallel = False, scale = emb_table.emb_scaling_factor) 
                    emb_table.embedding_grad_buffer.add_(buffer_changes) # buffer accumulates integer tensors, scales handles the batch size 
            else: 
                raise Warning("Cannot find the list of embedding tables") 
        else: 
            if model.emb_l is not None: 
                for emb_table in model.emb_l: 
                    emb_table.embedding_grad_buffer.add_(emb_table.embedding_bag.weight.grad/number_of_gpus) 
            else: 
                raise Warning("Cannot find the list of embedding tables") 
        '''
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
        ''' 
        
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad/number_of_gpus) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad/number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight_grad_buffer.add_(layer_one.weight.grad/number_of_gpus) 
                    layer_one.bias_grad_buffer.add_(layer_one.bias.grad/number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 

def grad_precision_and_scale(model, number_of_gpus, rank_for_debug, output_flag = False): 
    ''' 
    The function is first called inside the backward propagation, then followed by gradient update with ranking_range True, and then call weight update 
    The function that uses the magnitude of the gradients to determine bit width. 
    First, the function computes the maximum and minimum of gradients for each table, communicate them across GPUs. 
    Second, the function rank the magnitude and assign bit width 
    The bit width is determined by bottom 50% is set to 4-bit, 30% is set to 8-bit, and the rest is 32-bit 

    Parameters 
    ---------- 
    model: the model that is being trained 
    number_of_gpus: number of GPUs to be used 
    rank_for_debug: for debug printing only, used to identify the rank that need printing 

    Return: 
    ---------- 
    None 
    ''' 
    with torch.no_grad(): 
        range_list = [] 
        if model.emb_l is not None: 
            for id, emb_table in enumerate(model.emb_l): 
                if emb_table.embedding_bag.weight.grad.grad_fn is not None: 
                    emb_table.embedding_bag.weight.grad.detach_() 
                else: 
                    emb_table.embedding_bag.weight.grad.requires_grad_(False) 
                # finding scale 
                '''
                emb_table.embedding_bag.weight.grad.coalesce() 
                ''' 
                range_incomplete = finding_range_for_gradient(emb_table.embedding_bag.weight.grad.coalesce().values()) 
                # communicate the range bound across GPUs 
                range_incomplete.requires_grad_(False) 
                dist.all_reduce(range_incomplete, op = dist.ReduceOp.SUM) 
                range_incomplete.mul_(1./number_of_gpus) 
                emb_table.emb_scaling_factor = range_incomplete # emb_scaling_factor to quantize gradients used as a place to store range 
                '''
                if rank_for_debug == 0: 
                    print("table {}, gradient scale is {}".format(id, emb_table.emb_scaling_factor)) 
                ''' 

                n = 2 ** (4 - 1) - 1 
                # when ranking normalize the gradient min and max based on the weights min and max 
                '''
                print("table {}, rank {}, ebs {}".format(id, rank_for_debug, emb_table.eb_scaling_factor)) 
                ''' 
                range_list.append((range_incomplete/(emb_table.eb_scaling_factor * n)).item()) 

        if rank_for_debug == 0: 
            '''
            list_id = np.random.permutation(26) 
            ''' 
            '''
            range_list = torch.Tensor(range_list) 
            ''' 
            prob_l = range_list/(np.sum(range_list)) 
            '''
            prob_l = range_list.softmax(dim = 0).numpy() 
            ''' 
            list_id = np.random.choice(26, 26, replace = False, p = prob_l) 
            list_id = list_id[::-1] 
            '''
            print("range collected: {}".format(range_list)) 
            print("Probability: {}".format(prob_l)) 
            print("samplin: {}".format(list_id)) 
            ''' 
            if rank_for_debug == 0 and output_flag: 
                print("rank {} ranking from least wide range to the widest range {}".format(rank_for_debug, list_id)) 
            for j, id in enumerate(list_id): 
                # ascending order low precision to high precision list 
                if (j <= 8): 
                    # the 50% of the tables that have the smallest range 
                    model.emb_l[id].gradient_bit_width.zero_() 
                elif (j <= 22): 
                    # the 30% of the tables that have the middle range 
                    model.emb_l[id].gradient_bit_width.zero_().add_(8) 
                else: 
                    # the 20% of the tables that have the large range 
                    model.emb_l[id].gradient_bit_width.zero_().add_(32) 
        dist.barrier() 
        
        # record the scale for quantizing gradients 
        for id, emb_table in enumerate(model.emb_l): 
            '''
            if rank_for_debug == 0: 
                print("table {}, gradient precision {}bit".format(id, emb_table.gradient_bit_width.item())) 
            ''' 
            dist.broadcast(model.emb_l[id].gradient_bit_width, 0) 
            '''
            if rank_for_debug == 0: 
                print("rank {}, table {}, gradient precision set to {}".format(rank_for_debug, id, emb_table.gradient_bit_width)) 
            ''' 
            if emb_table.gradient_bit_width == 0: 
                continue 
            if emb_table.gradient_bit_width == 32: 
                continue 
            n = 2 ** (emb_table.gradient_bit_width - 1) - 1 
            emb_table.emb_scaling_factor = torch.clamp(emb_table.emb_scaling_factor, min = 1e-8) / n 

def grad_update_parallel_comm(model, number_of_gpus, emb_grad_quantized = True, num_bits = 16, ranking_range = False, rank_for_debug = None, iteration_count = None, mlp_layer_quantized = True): 
    ''' 
    The function quantize and synchronize the gradients 

    Parameters: 
    ---------- 
    model: the model that is being trained 
    number_of_gpus: number of gpus in the system 
    emb_grad_quantized: flag to check whether to quantize embedding tables 
    num_bits: number of bits to use when emb_grad_quantized is true, otherwise None 
    ranking_range: bit width not assigned uniformly but based on their magnitude 
    rank_for_debug: rank of the machine 

    Return: 
    ---------- 
    None 
    '''
    with torch.no_grad(): 
        # only embedding gradient to be quantized do gradient changes needed, if not quantized, the gradient doesn't need to be changed 
        if emb_grad_quantized: 
            if model.emb_l is not None: 
                for id, emb_table in enumerate(model.emb_l): 
                    # skip tables that don't need update 
                    if ranking_range and emb_table.gradient_bit_width.item() == 0: 
                        '''
                        print("rank {}, table {}, gradient precision set to {}".format(rank_for_debug, id, emb_table.gradient_bit_width)) 
                        ''' 
                        continue 
                    if ranking_range and emb_table.gradient_bit_width.item() == 32: 
                        '''
                        print("rank {}, table {}, gradient precision set to {}".format(rank_for_debug, id, emb_table.gradient_bit_width)) 
                        ''' 
                        continue 
                    if not ranking_range: 
                        buffer_changes, scale = quantize_emb_grad(embedding_table = None, embedding_table_grad = emb_table.embedding_bag.weight.grad, num_bits = num_bits, parallel = True, num_gpus = number_of_gpus, table_id = id) 
                        '''
                        if rank_for_debug == 0: 
                            print(buffer_changes.dtype) 
                        ''' 
                        emb_table.emb_scaling_factor = scale 
                    else: 
                        '''
                        print("rank {}, table {}, gradient precision set to {}".format(rank_for_debug, id, emb_table.gradient_bit_width)) 
                        ''' 
                        buffer_changes, _ = quantize_emb_grad_two(emb_table, number_of_gpus) 
                        # clear grad to be zero 
                    if emb_table.embedding_bag.weight.grad_fn is not None: 
                        emb_table.embedding_bag.weight.grad.detach() 
                    else: 
                        emb_table.embedding_bag.weight.grad.requires_grad_(False) 
                    emb_table.embedding_bag.weight.grad.zero_() 
                    # set grad to be the quantized value 
                    emb_table.embedding_bag.weight.grad.add_(buffer_changes) 
                    '''
                    if count == 0: 
                        emb_table.embedding_bag.weight.grad.coalesce() 
                        print(emb_table.embedding_bag.weight.grad.values()[0]) 
                    count += 1 
                    ''' 
            else: 
                raise Warning("Cannot find the list of embedding tables") 
        else: 
            if model.emb_l is not None: 
                for emb_table in model.emb_l: 
                    if emb_table.embedding_bag.weight.grad_fn is not None: 
                        emb_table.embedding_bag.weight.grad.detach() 
                    else: 
                        emb_table.embedding_bag.weight.grad.requires_grad_(False) 
                    emb_table.embedding_bag.weight.grad.coalesce() 
                    dist.all_reduce(emb_table.embedding_bag.weight.grad, dist.ReduceOp.SUM) 
                    emb_table.embedding_bag.weight.grad.mul_(1. / number_of_gpus) 
            else: 
                raise Warning("Cannot find the list of embedding tables") 
        '''
        if rank_for_debug == 0 and (iteration_count + 1) % 1024 == 0: 
            global total_comm_time 
            print("total time per iteration spent on all reduce is {} ms".format([l * 1000.0/1024 for l in total_comm_time])) 
            print(np.sum(total_comm_time) * 1000.0/1024) 
            total_comm_time = [0 for k in range(26)] 
        ''' 
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    if mlp_layer_quantized: 
                        buffer_changes, scale = quantize_linear_grad(layer_one, num_bits = 8, parallel = True, num_gpus = number_of_gpus, err_compensation = False) 
                        layer_one.weight_scaling_factor = scale 
                        if layer_one.weight.grad.grad_fn is not None: 
                            layer_one.weight.grad.detach() 
                        else: 
                            layer_one.weight.grad.requires_grad_(False) 
                        layer_one.weight.grad.zero_() 
                        layer_one.weight.grad.add_(buffer_changes) 
                        
                        buffer_changes, scale = quantize_bias_grad(layer_one, num_bits = 8, parallel = True, num_gpus = number_of_gpus, err_compensation = False) 
                        layer_one.bias_scaling_factor = scale 
                        if layer_one.bias.grad.grad_fn is not None: 
                            layer_one.bias.grad.detach() 
                        else: 
                            layer_one.bias.grad.requires_grad_(False) 
                        layer_one.bias.grad.zero_() 
                        layer_one.bias.grad.add_(buffer_changes) 
                    else: 
                        if layer_one.weight.grad.grad_fn is not None: 
                            layer_one.weight.grad.detach() 
                        else: 
                            layer_one.weight.grad.requires_grad_(False) 
                        dist.all_reduce(layer_one.weight.grad, dist.ReduceOp.SUM) 
                        layer_one.weight.grad.mul_(1. / number_of_gpus) 
                        if layer_one.bias.grad.grad_fn is not None: 
                            layer_one.bias.grad.detach() 
                        else: 
                            layer_one.bias.grad.requires_grad_(False) 
                        dist.all_reduce(layer_one.bias.grad, dist.ReduceOp.SUM) 
                        layer_one.bias.grad.mul_(1. / number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 

        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    if mlp_layer_quantized: 
                        buffer_changes, scale = quantize_linear_grad(layer_one, num_bits = 8, parallel = True, num_gpus = number_of_gpus, err_compensation = False) 
                        layer_one.weight_scaling_factor = scale 
                        if layer_one.weight.grad.grad_fn is not None: 
                            layer_one.weight.grad.detach() 
                        else: 
                            layer_one.weight.grad.requires_grad_(False) 
                        layer_one.weight.grad.zero_() 
                        layer_one.weight.grad.add_(buffer_changes) 

                        buffer_changes, scale = quantize_bias_grad(layer_one, num_bits = 8, parallel = True, num_gpus = number_of_gpus, err_compensation = False) 
                        layer_one.bias_scaling_factor = scale 
                        if layer_one.bias.grad.grad_fn is not None: 
                            layer_one.bias.grad.detach() 
                        else: 
                            layer_one.bias.grad.requires_grad_(False) 
                        layer_one.bias.grad.zero_() 
                        layer_one.bias.grad.add_(buffer_changes) 
                    else: 
                        if layer_one.weight.grad.grad_fn is not None: 
                            layer_one.weight.grad.detach() 
                        else: 
                            layer_one.weight.grad.requires_grad_(False) 
                        dist.all_reduce(layer_one.weight.grad, dist.ReduceOp.SUM) 
                        layer_one.weight.grad.mul_(1. / number_of_gpus) 
                        if layer_one.bias.grad.grad_fn is not None: 
                            layer_one.bias.grad.detach() 
                        else: 
                            layer_one.bias.grad.requires_grad_(False) 
                        dist.all_reduce(layer_one.bias.grad, dist.ReduceOp.SUM) 
                        layer_one.bias.grad.mul_(1. / number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        '''
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    if layer_one.weight.grad.grad_fn is not None: 
                        layer_one.weight.grad.detach() 
                    else: 
                        layer_one.weight.grad.requires_grad_(False) 
                    dist.all_reduce(layer_one.weight.grad, dist.ReduceOp.SUM) 
                    layer_one.weight.grad.mul_(1. / number_of_gpus) 
                    if layer_one.bias.grad.grad_fn is not None: 
                        layer_one.bias.grad.detach() 
                    else: 
                        layer_one.bias.grad.requires_grad_(False) 
                    dist.all_reduce(layer_one.bias.grad, dist.ReduceOp.SUM) 
                    layer_one.bias.grad.mul_(1. / number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    if layer_one.weight.grad.grad_fn is not None: 
                        layer_one.weight.grad.detach() 
                    else: 
                        layer_one.weight.grad.requires_grad_(False) 
                    dist.all_reduce(layer_one.weight.grad, dist.ReduceOp.SUM) 
                    layer_one.weight.grad.mul_(1. / number_of_gpus) 
                    if layer_one.bias.grad.grad_fn is not None: 
                        layer_one.bias.grad.detach() 
                    else: 
                        layer_one.bias.grad.requires_grad_(False) 
                    dist.all_reduce(layer_one.bias.grad, dist.ReduceOp.SUM) 
                    layer_one.bias.grad.mul_(1. / number_of_gpus) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 
        ''' 

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
            if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
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
            if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
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
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight.data.add_(-lr * layer_one.weight_grad_buffer) 
                    layer_one.bias.data.add_(-lr * layer_one.bias_grad_buffer) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight.data.add_(-lr * layer_one.weight_grad_buffer) 
                    layer_one.bias.data.add_(-lr * layer_one.bias_grad_buffer) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 

def weights_update_added_quantization(model, lr, num_gpus, emb_grad_quantized = True, update_embedding = True): 
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
        if emb_grad_quantized and update_embedding: 
            if model.emb_l is not None: 
                for emb_table in model.emb_l: 
                    weight_update = emb_table.embedding_grad_buffer * (emb_table.emb_scaling_factor/num_gpus) # dequantize 
                    emb_table.embedding_bag.weight.data.add_(-lr * weight_update) # update 
            else: 
                raise Warning("Cannot find the list of embedding tables") 
        elif update_embedding: 
            if model.emb_l is not None: 
                for emb_table in model.emb_l: 
                    emb_table.embedding_bag.weight.data.add_(-lr * emb_table.embedding_grad_buffer) 
            else: 
                raise Warning("Cannot find the list of embedding tables") 
        
        '''
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
        ''' 
        
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight.data.add_(-lr * layer_one.weight_grad_buffer) 
                    layer_one.bias.data.add_(-lr * layer_one.bias_grad_buffer) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight.data.add_(-lr * layer_one.weight_grad_buffer) 
                    layer_one.bias.data.add_(-lr * layer_one.bias_grad_buffer) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 

def weight_update_parallel_comm(model, lr, emb_grad_quantized = True, update_embedding = True, num_gpus = 1, rank_for_debug = None, ranking_range = False, use_ec = False, mlp_layer_quantized = True): 
    with torch.no_grad(): 
        if update_embedding: 
            if model.emb_l is not None: 
                for id, emb_table in enumerate(model.emb_l): 
                    if emb_grad_quantized: 
                        '''
                        print("rank {} table{} first quantized in integer or ratio then s, bit width {}".format(rank_for_debug, id, emb_table.gradient_bit_width.item())) 
                        ''' 
                        if ranking_range and emb_table.gradient_bit_width.item() == 0: 
                            '''
                            print("rank {} table {}, weights gradient bit width is 0 and not updating".format(rank_for_debug, id)) 
                            ''' 
                            continue 
                        if ranking_range and emb_table.gradient_bit_width.item() == 32: 
                            emb_table.embedding_bag.weight.data.add_(-lr * emb_table.embedding_bag.weight.grad) 
                        else: 
                            grad_update_n = emb_table.embedding_bag.weight.grad * emb_table.emb_scaling_factor.item() 
                            '''
                            emb_table.embedding_bag.weight.data.add_(-lr * emb_table.embedding_bag.weight.grad * emb_table.emb_scaling_factor.item()) 
                            ''' 
                            emb_table.embedding_bag.weight.data.add_(-lr * grad_update_n) 
                            if use_ec: 
                                update_error_term(emb_table, grad_update_n) 
                    else: 
                        emb_table.embedding_bag.weight.data.add_(-lr * emb_table.embedding_bag.weight.grad) 
            else: 
                raise Warning("Cannot find the list of embedding tables") 
        
        if model.bot_l is not None: 
            for i, layer_one in enumerate(model.bot_l): 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    if mlp_layer_quantized: 
                        '''
                        if rank_for_debug == 0: 
                            print("layer {}".format(i)) 
                            print(layer_one.weight.grad[0]) 
                            print(layer_one.weight_scaling_factor) 
                            print(layer_one.bias.grad) 
                            print(layer_one.bias_scaling_factor) 
                        ''' 
                        layer_one.weight.data.add_(-lr * layer_one.weight.grad * layer_one.weight_scaling_factor.view(-1, 1)) 
                        layer_one.bias.data.add_(-lr * layer_one.bias.grad * layer_one.bias_scaling_factor) 
                    else: 
                        layer_one.weight.data.add_(-lr * layer_one.weight.grad) 
                        layer_one.bias.data.add_(-lr * layer_one.bias.grad) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        
        if model.top_l is not None: 
            for i, layer_one in enumerate(model.top_l): 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    if mlp_layer_quantized: 
                        '''
                        if rank_for_debug == 0: 
                            print("layer {}".format(i)) 
                            print(layer_one.weight.grad[0]) 
                            print(layer_one.weight_scaling_factor) 
                            print(layer_one.bias.grad) 
                            print(layer_one.bias_scaling_factor) 
                        ''' 
                        layer_one.weight.data.add_(-lr * layer_one.weight.grad * layer_one.weight_scaling_factor.view(-1, 1)) 
                        layer_one.bias.data.add_(-lr * layer_one.bias.grad * layer_one.bias_scaling_factor) 
                    else: 
                        layer_one.weight.data.add_(-lr * layer_one.weight.grad) 
                        layer_one.bias.data.add_(-lr * layer_one.bias.grad) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 
        '''
        if model.bot_l is not None: 
            for layer_one in model.bot_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight.data.add_(-lr * layer_one.weight.grad) 
                    layer_one.bias.data.add_(-lr * layer_one.bias.grad) 
        else: 
            raise Warning("Cannot find the list of bottom linear layers") 
        
        if model.top_l is not None: 
            for layer_one in model.top_l: 
                if isinstance(layer_one, QuantLinear) or isinstance(layer_one, LinearCompressedGrad): 
                    layer_one.weight.data.add_(-lr * layer_one.weight.grad) 
                    layer_one.bias.data.add_(-lr * layer_one.bias.grad) 
        else: 
            raise Warning("Cannot find the list of top linear layers") 
        ''' 

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

def update_sparse_tensor(sparse_m, list_idx): 
    ''' 
    This function output list of indices relative to the sparse matrix and the absolute row ids list 
    both need to be used 

    Parameter: 
    ---------- 
    sparse_m: update 
    list_idx: index interest 

    Return: 
    ---------- 
    list_ll: list of relative ids 
    ids_row: list of absolute ids 
    '''
    sparse_m = sparse_m.coalesce() 
    list_ll = [] 
    ids_row = [] 
    trace_count = 0 
    list_indices2 = sparse_m.indices()[0] 
    for idx in list_idx: 
        while (trace_count < len(list_indices2)):  
            ele = list_indices2[trace_count] 
            if ele.item() == idx: 
                list_ll.append(trace_count) 
                ids_row.append(ele.item()) 
                break 
            trace_count += 1 
        if (trace_count >= len(list_indices2)): 
            break 
    return list_ll, ids_row 

def update_error_term(embedding_table, sparse_grad): 
    ''' 
    This function requires embedding_table to have a field called index_interest and a field called sparse_error 
    Also, embedding_table.index_interest is sorted from small to large 

    Parameter: 
    ---------- 
    embedding_table: the embedding table that is to have the quantized gradients 
    sparse_error: error term that is updating 

    Return: 
    ---------- 
    None 
    ''' 
    device = embedding_table.sparse_error.device 
    relative_ids, abs_ids = update_sparse_tensor(sparse_grad, embedding_table.index_interest) 
    error_update = torch.sparse_coo_tensor(torch.Tensor(abs_ids, device = device), sparse_grad.values()[relative_ids], size = embedding_table.sparse_error.size, device = device) 
    embedding_table.sparse_error -= error_update 
    return 

def quantize_emb_grad_error_compensation(embedding_table, num_gpus = None, ranking_range = False, num_bits = None): 
    ''' 
    This function requires embedding_table to have a field called sparse_error 
    
    The local error compensation is implemented using the error term E 
    Z = T_g + E 
    T_acc = int_quantize(Z) 
    E = Z - T_acc 
    output (T_acc, E) 

    Parameter: 
    --------- 
    embedding_table: the embedding table that is to have the quantized gradients 
    num_gpus: number of GPUs is used in the system 
    ranking_range: a flag determining whether ranking range is supported 
    num_bits: number of bits are used to quantize the embedding table 

    Return: 
    ---------- 
    None 
    '''
    with torch.no_grad(): 
        
        # TODO remove assert 
        assert(ranking_range == False) 

        if embedding_table.embedding_bag.weight.grad.grad_fn is not None: 
            embedding_table.embedding_bag.weight.grad.detach_() 
        else: 
            embedding_table.embedding_bag.weight.grad.requires_grad_(False) 
        
        if not ranking_range: 
            embedding_table = embedding_table.coalesce() 
            if scale is None: 
                scale = symmetric_linear_quantization_param_two(num_bits, embedding_table.values(), None, None, None) 

                scale.requires_grad_(False) 
                dist.all_reduce(scale, dist.ReduceOp.SUM) 
                scale = scale/num_gpus 
            scale = scale.view(-1) 
            # quantize 
            emb_gradient_update = torch.sparse_coo_tensor(embedding_table.indices(), SymmetricQuantFunction.apply(embedding_table.values(), num_bits, scale), size = embedding_table.size(), device = embedding_table.device) 

            emb_gradient_update.requires_grad_(False) 
            dist.all_reduce(emb_gradient_update, dist.ReduceOp.SUM) 
            emb_gradient_update.mul_(1. / num_gpus) 
            return emb_gradient_update, scale 

def quantize_emb_grad_two(embedding_table, num_gpus = None): 
    with torch.no_grad(): 
        if embedding_table.embedding_bag.weight.grad.grad_fn is not None: 
            embedding_table.embedding_bag.weight.grad.detach_() 
        else: 
            embedding_table.embedding_bag.weight.grad.requires_grad_(False) 
        
        # quantize 
        emb_gradient_update = torch.sparse_coo_tensor(embedding_table.embedding_bag.weight.grad.coalesce().indices(), SymmetricQuantFunction.apply(embedding_table.embedding_bag.weight.grad.coalesce().values(), embedding_table.gradient_bit_width, embedding_table.emb_scaling_factor), size = embedding_table.embedding_bag.weight.grad.size(), device = embedding_table.embedding_bag.weight.grad.device) 
        emb_gradient_update.requires_grad_(False) 
        dist.all_reduce(emb_gradient_update, dist.ReduceOp.SUM) 
        emb_gradient_update.mul_(1. / num_gpus) 
        return emb_gradient_update, None 

def quantize_emb_grad(embedding_table, embedding_table_grad, num_bits, parallel, num_gpus = None, scale = None, use_ec = False, table_id = None): 
    with torch.no_grad(): 
        if embedding_table_grad.grad_fn is not None: 
            embedding_table_grad.detach_() 
        else: 
            embedding_table_grad.requires_grad_(False) 
        # finding scale 
        if use_ec: 
            embedding_table_grad += embedding_table.sparse_error 
        embedding_table_grad = embedding_table_grad.coalesce() 
        if scale is None: 
            scale = symmetric_linear_quantization_param_two(num_bits, embedding_table_grad.values(), None, None, None) 

        if parallel: 
            scale.requires_grad_(False) 
            dist.all_reduce(scale, dist.ReduceOp.SUM) 
            scale.mul_(1. / (num_gpus)) 
        scale = scale.view(-1) 
        # quantize 
        emb_gradient_update = torch.sparse_coo_tensor(embedding_table_grad.indices(), SymmetricQuantFunction.apply(embedding_table_grad.values(), num_bits, scale), size = embedding_table_grad.size(), device = embedding_table_grad.device) 
        '''
        emb_gradient_update = torch.sparse_coo_tensor(embedding_table_grad.indices(), SymmetricQuantFunction.apply(embedding_table_grad.values(), num_bits, scale).type(torch.int8), size = embedding_table_grad.size(), device = embedding_table_grad.device) 
        ''' 
        if parallel: 
            emb_gradient_update.requires_grad_(False) 
            '''
            before_a = time_wrap(True) 
            ''' 
            dist.all_reduce(emb_gradient_update, dist.ReduceOp.SUM) 
            '''
            after_a = time_wrap(True) 
            if table_id != None: 
                global total_comm_time 
                total_comm_time[table_id] = after_a - before_a 
            ''' 
            emb_gradient_update.mul_(1. / num_gpus) 
            '''
            emb_gradient_update = emb_gradient_update.coalesce() 
            emb_gradient_update = torch.sparse_coo_tensor(emb_gradient_update.indices(), emb_gradient_update.values().round_(), size = emb_gradient_update.size(), device = emb_gradient_update.device) 
            ''' 
        return emb_gradient_update, scale 

def quantize_linear_grad(layer, num_bits, parallel, num_gpus = None, per_channel = True, scale = None, err_compensation = False): 
    with torch.no_grad(): 
        if layer.weight.grad.grad_fn is not None: 
            layer.weight.grad.detach_() 
        else: 
            layer.weight.grad.requires_grad_(False) 
        
        if err_compensation: 
            weight = layer.weight.grad + layer.error_compensation_weight 
        else: 
            weight = layer.weight.grad 
        
        if scale is None: 
            # finding scale 
            if per_channel: 
                w_min, _ = torch.min(weight, dim = 1, out = None) 
                w_max, _ = torch.max(weight, dim = 1, out = None) 
            else: 
                w_min = weight.min().expand(1) 
                w_max = weight.max().expand(1) 
            fc_scaling_factor = symmetric_linear_quantization_params(num_bits, w_min, w_max, per_channel) 
        else: 
            fc_scaling_factor = scale 
        if parallel: 
            dist.all_reduce(fc_scaling_factor, dist.ReduceOp.SUM) 
            fc_scaling_factor.mul_(1. / num_gpus) 
        # quantize 
        grad_up = SymmetricQuantFunction.apply(weight, num_bits, fc_scaling_factor) 
        
        if parallel: 
            grad_up.requires_grad_(False) 
            dist.all_reduce(grad_up, dist.ReduceOp.SUM) 
            grad_up.mul_(1. / num_gpus) 

        if err_compensation: 
            layer.error_compensation_weight = weight - (grad_up * fc_scaling_factor.view(-1, 1))  
        
        return grad_up, fc_scaling_factor 

def quantize_bias_grad(layer, num_bits, parallel, num_gpus = None, scale = None, err_compensation = False): 
    with torch.no_grad(): 
        if layer.bias.grad.grad_fn is not None: 
            layer.bias.grad.detach_() 
        else: 
            layer.bias.grad.requires_grad_(False) 
        
        if err_compensation: 
            bias = layer.bias.grad + layer.error_compensation_bias 
        else: 
            bias = layer.bias.grad 
        
        if scale is None: 
            # finding scale 
            bias_min, _ = torch.min(bias, dim = 0, out = None) 
            bias_max, _ = torch.max(bias, dim = 0, out = None) 
            scale = symmetric_linear_quantization_params(num_bits, bias_min, bias_max) 
        if parallel: 
            dist.all_reduce(scale, dist.ReduceOp.SUM) 
            scale.mul_(1. / num_gpus) 
        # quantize 
        grad_update = SymmetricQuantFunction.apply(bias, num_bits, scale) 
        if parallel: 
            grad_update.requires_grad_(False) 
            dist.all_reduce(grad_update, dist.ReduceOp.SUM) 
            grad_update.mul_(1. / num_gpus) 

        if err_compensation: 
            layer.error_compensation_bias = bias - (grad_update * scale) 
        
        return grad_update, scale 
        
def weight_syncc(dlrm, num_gpus): 
    with torch.no_grad(): 
        model = dlrm 
        for name, param in model.named_parameters(): 
            param.requires_grad_(False) 
            dist.all_reduce(param, dist.ReduceOp.SUM) 
            param.mul_(1. / num_gpus) 
            param.requires_grad_(True) 
