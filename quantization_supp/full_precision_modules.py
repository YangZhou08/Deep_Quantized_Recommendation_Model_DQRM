import torch 
import time 
import math 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.multiprocessing as mp 
from torch.nn import Module, Parameter 
from .quant_utils import * 

class EmbeddingBagCompressedGrad(Module): 
    def __init__(self, 
                num_embeddings, 
                embedding_dim, 
                gradient_precision = 8): 
        super(EmbeddingBagCompressedGrad, self).__init__() 
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim 
        self.gradient_precision = gradient_precision 

        W = np.random.uniform(
            low = -np.sqrt(1/self.num_embeddings), high = np.sqrt(1/self.num_embeddings), size = (self.num_embeddings, self.embedding_dim) 
        ).astype(np.float32) 

        self.register_buffer('embedding_grad_buffer', torch.zeros((self.num_embeddings, self.embedding_dim)), persistent = False) 
        self.register_buffer('emb_scaling_factor', torch.zeros(1), persistent = True) 

        self.embedding_bag = nn.EmbeddingBag(self.num_embeddings, self.embedding_dim, mode = "sum", sparse = True) 
        self.embedding_bag.weight.data = torch.tensor(W, requires_grad = True) 

    def __repr__(self): 
        s = super(EmbeddingBagCompressedGrad, self).__repr__() 
        s = "(" + s + " gradient compression information precision: {})".format(self.gradient_precision) 

    def forward(self, input, offsets, per_sample_weights): 
        return self.embedding_bag(input, offsets, per_sample_weights = None) 

class LinearCompressedGrad(Module): 
    def __init__(self, gradient_precision): 
        self.gradient_precision = gradient_precision 
        
    def set_param(self, linear): 
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone()) 

        self.register_buffer('weight_grad_buffer', torch.zeros_like(self.weight), persistent = False) 
        self.register_buffer('weight_scaling_factor', torch.zeros(self.out_features)) 

        self.register_buffer('bias_grad_buffer', torch.zeros(self.out_features)) 
        self.register_buffer('bias_scaling_factor', torch.zeros(self.out_features)) 
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None 
    
    def forward(self, x, prev_act_scaling_factor = None): 
        return F.linear(x, weight = self.weight, bias = self.bias) 
