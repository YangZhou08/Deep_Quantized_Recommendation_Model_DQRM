import torch as torch
from .quantizer import lsq 
import torch.nn.functional as F 
import torch.nn as nn 
import numpy as np 
from torch.nn import Parameter 

class QuantConv2dLSQ(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        output = self.quan_a_fn(x)
        return self._conv_forward(output, quantized_weight) 


class QuantLinearLSQ(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn = lsq.LsqQuan, quan_a_fn = lsq.LsqQuan): 
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn(bit=4, all_positive=False, symmetric=False, per_channel = True) 
        self.quan_a_fn = quan_a_fn(bit=4, all_positive=False, symmetric=False, per_channel = True) 

        self.weight = nn.Parameter(m.weight.detach())
        print("weight shape: ", self.weight.shape) 
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach()) 
        else: 
            self.bias = None 

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight) 
        print("quantized weight shape: ", quantized_weight.shape) 
        quantized_bias = self.quan_w_fn(self.bias) 
        print("input size: ", x.shape) 
        # output = self.quan_a_fn(x) 
        return F.linear(x, quantized_weight, quantized_bias) 

'''
QuanModuleMapping = {
    torch.nn.Conv2d: QuantConv2dLSQ,
    torch.nn.Linear: QuantLinearLSQ
}
''' 

class QuantEmbeddingBagLSQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, full_precision_flag=False, embedding_id=None, quan_w_fn=lsq.LsqQuan, quan_a_fn=lsq.LsqQuan):
        super(QuantEmbeddingBagLSQ, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode="sum", sparse=True)

        W = np.random.uniform(
            low=-np.sqrt(1 / self.num_embeddings),
            high=np.sqrt(1 / self.num_embeddings),
            size=(self.num_embeddings, self.embedding_dim)
        ).astype(np.float32)
        self.embedding_bag.weight.data = torch.tensor(W, requires_grad=True)

        self.quan_w_fn = quan_w_fn(bit=4, all_positive=False, symmetric=False, per_channel=False) 
        self.quan_a_fn = quan_a_fn(bit=4, all_positive=False, symmetric=False, per_channel=True) 

        self.quan_w_fn.init_from(self.embedding_bag.weight)

    def forward(self, input, offsets=None, full_precision_flag = False, per_sample_weights=None, test_mode = False):
        if not full_precision_flag:
            '''
            quantized_weight = self.quan_w_fn(self.embedding_bag.weight)
            ''' 
            '''
            output = self.embedding_bag(input, weight = quantized_weight, offsets = offsets, sparse = True, mode = "sum") # right now,
            ''' 
            output = self.embedding_bag(input, offsets, per_sample_weights = None) 
            output = self.quan_w_fn(output) 
            # output = self.quan_a_fn(output) 
        else:
            output = F.embedding_bag(input, weight = self.embedding_bag.weight, offsets = offsets, sparse = True, mode = "sum") # right now,
            return output 
        return output 
