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

def quantized_gradients_update(model, arg, lr): 
    # no weight decay is used 
    world_size = float(arg.world_size) 
    for name, param in model.named_parameters(): 
        update = param.grad.data # finding the gradient of the data by layer 
        dist.all_reduce(update, op=dist.ReduceOp.SUM) 
        param.data.add_(lr, update) 
        '''
        param.grad.data *= 0 
        ''' 

def clear_gradients(model): 
    for name, param in model.named_parameters(): 
        print(name) 
        if param.grad.data is not None: 
            param.grad.data *= 0 
