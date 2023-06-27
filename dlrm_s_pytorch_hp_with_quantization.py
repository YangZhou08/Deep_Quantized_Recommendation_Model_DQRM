# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
import dlrm_data_pytorch as dp

# For distributed run
import extend_distributed_three as ext_dist_three 
import mlperf_logger

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
import optim.rwsadagrad as RowWiseSparseAdagrad
from torch.utils.tensorboard import SummaryWriter

# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag

from quantization_supp.quant_modules_not_quantize_grad import QuantEmbeddingBagTwo 
from quantization_supp.quant_modules_not_quantize_grad import QuantLinear 
from quantization_supp.quant_modules_not_quantize_grad import QuantAct 
from quantization_supp.quant_utils import symmetric_linear_quantization_params 
from quantization_supp.quant_utils import SymmetricQuantFunction 
from quantization_supp.quant_utils import linear_quantize 

from sgd_quantized_gradients import quantized_gradients_update 
from sgd_quantized_gradients import clear_gradients 

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print("Unable to import onnx. ", error)

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

exc = getattr(builtins, "IOError", "FileNotFoundError")


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, ndevices=1): 
    '''
    print("Calling inside dlrm wrap: ndevices is ", ndevices) 
    ''' 
    with record_function("DLRM forward"):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            if ndevices == 1:
                lS_i = (
                    [S_i.to(device) for S_i in lS_i]
                    if isinstance(lS_i, list)
                    else lS_i.to(device)
                )
                lS_o = (
                    [S_o.to(device) for S_o in lS_o]
                    if isinstance(lS_o, list)
                    else lS_o.to(device)
                )
        return dlrm(X.to(device), lS_o, lS_i)


def loss_fn_wrap(Z, T, use_gpu, device):
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            return dlrm.loss_fn(Z, T.to(device))
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
            loss_fn_ = dlrm.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer, quant_linear_layer = False, channelwise_lin = False, quantize_activation = False): 
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # print("rank {} mlp layer with dimension {} and or by {}".format(ext_dist_two.my_local_rank, n, m)) 
            print("rank {} mlp layer with dimension {} and or by {}".format(ext_dist_three.my_local_rank, n, m)) 
            # construct fully connected operator 
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32) 
            '''
            LL = nn.Linear(int(n), int(m), bias = False) 
            mean = 0.0 
            std_dev = np.sqrt(2 / (m + n)) 
            W = np.random.normal(mean, std_dev, size = (m, n)).astype(np.float32) 
            ''' 
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True) 
            LL.bias.data = torch.tensor(bt, requires_grad=True) 
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True) 
            if self.quantization_flag and quant_linear_layer: # TODO recheck intentionally reverse logic updated: checked 
                '''
                print("use quant linear, input {}, output {}, quantization bit width {}, use full precision {}".format(n, m, self.weight_bit, "32-bit single precision" if not self.quantize_act_and_lin else "quantized")) 
                ''' 
                print("use quant linear, input {}, output {}, quantization bit width {}, use full precision {} and channelwise status {}".format(n, m, self.weight_bit, "32-bit single precision" if not self.quantize_act_and_lin else "quantized", "channelwise" if self.channelwise_lin else "not channelwise")) 
                QuantLnr = QuantLinear( 
                    weight_bit = self.weight_bit, 
                    bias_bit = self.weight_bit, 
                    full_precision_flag = not self.quantize_act_and_lin, 
                    per_channel = channelwise_lin, 
                    quantize_activation = quantize_activation 
                ) 
                QuantLnr.set_param(LL) 
                layers.append(QuantLnr) 
            else: 
                layers.append(LL) 
            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid()) 
            else:
                layers.append(nn.ReLU()) 

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers 
        if not self.quantization_flag: # TODO recheck intentionally reversed logic updated: checked 
            return torch.nn.Sequential(*layers) 
        else: 
            return layers 
        '''
        return layers 
        ''' 

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = [] 
        print("world size found is " + str(ext_dist_three.my_size)) 
        for i in range(0, ln.size): 
            # each gpu will have all embedding tables 
            # if ext_dist_two.my_size > 1: 
            if ext_dist_three.my_size > 1: 
                if i not in self.local_emb_indices:
                    continue 
            n = ln[i]

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            elif self.quantization_flag: 
                print("---------- Embedding Table {}, quantization used, n = {}, m = {}, quantization bit set to {}".format(i, n, m, self.embedding_bit)) 
                EE = QuantEmbeddingBagTwo(n, m, self.embedding_bit, embedding_id = i) 
            else:
                # print("---------- rank {} Embedding Table {}, quantization not used, n = {}, m = {}".format(ext_dist_two.my_rank, i, n, m)) 
                print("---------- rank {} Embedding Table {}, quantization not used, n = {}, m = {}".format(ext_dist_three.my_rank, i, n, m)) 
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True) 
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n)) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32) 
                '''
                W = np.random.normal(
                    loc = 0, scale = np.sqrt(1/n), size = (n, m) 
                ).astype(np.float32) 
                ''' 
                '''
                W = np.random.normal( 
                    loc = 0, scale = 0.03, size = (n, m) 
                ).astype(np.float32) 
                ''' 
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True) 
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce", 
        quantization_flag = False, 
        embedding_bit = 32, 
        modify_feature_interaction = False, 
        weight_bit = 8, 
        quantize_act_and_lin = False, 
        mlp_channelwise = False, 
        quantize_activation = False, 
        deviceid = None, 
        args = None): 
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function=loss_function 
            
            # add quantization identification 
            self.quantization_flag = quantization_flag 
            self.embedding_bit = embedding_bit 
            self.modify_feature_interaction = modify_feature_interaction 
            self.weight_bit = weight_bit 
            self.quantize_act_and_lin = quantize_act_and_lin and quantization_flag # only if both is true 
            self.change_lin_from_full_to_quantized = False 
            self.channelwise_lin = mlp_channelwise 
            self.quantize_activation = quantize_activation 
            self.deviceid = deviceid 
            self.args = args 

            if self.quantization_flag: 
                self.quant_input = QuantAct(activation_bit = self.weight_bit if self.weight_bit >= 8 else 8, act_range_momentum = -1) 
                self.quant_feature_outputs = QuantAct(fixed_point_quantization = True, activation_bit = self.weight_bit if self.weight_bit >= 8 else 8, act_range_momentum = -1) # recheck activation_bit 
                self.register_buffer('feature_xmin', torch.zeros(1)) 
                self.register_buffer('feature_xmax', torch.zeros(1)) 
                self.register_buffer('features_scaling_factor', torch.zeros(1)) 
            
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # If running distributed, get local slice of embedding tables
            # if ext_dist.my_size > 1:
            # if ext_dist_two.my_size > 1: 
            if ext_dist_three.my_size > 1: 
                n_emb = len(ln_emb)
                # if n_emb < ext_dist.my_size:
                # if n_emb < ext_dist_two.my_size: 
                if n_emb < ext_dist_three.my_size: 
                    sys.exit(
                        "only (%d) sparse features for (%d) devices, table partitions will fail"
                        % (n_emb, args.world_size) # (n_emb, ext_dist.my_size)
                    )
                self.n_global_emb = n_emb
                # self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths( 
                '''
                self.n_local_emb, self.n_emb_per_rank = ext_dist_two.get_split_lengths( 
                    n_emb
                ) 
                ''' 
                self.n_local_emb, self.n_emb_per_rank = ext_dist_three.get_split_lengths(
                    n_emb 
                )
                # self.local_emb_slice = ext_dist.get_my_slice(n_emb) 
                # self.local_emb_slice = ext_dist_two.get_my_slice(n_emb) 
                self.local_emb_slice = ext_dist_three.get_my_slice(n_emb) 
                self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

                # print("rank {}, local embedding indices {}".format(ext_dist_two.my_rank, self.local_emb_indices)) 
                print("rank {}, local embedding indices {}".format(ext_dist_three.my_rank, self.local_emb_indices)) 

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling) 
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list 
            '''
            else: 
                self.v_W_l = [] 
                for i in range(len(ln_emb)): 
                    n = ln_emb[i] 
                    self.v_W_l.append(Parameter(torch.ones(n, dtype = torch.float32))) 
            ''' 
            '''
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot) 
            ''' 
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot, quant_linear_layer = self.quantize_act_and_lin, channelwise_lin = self.channelwise_lin, quantize_activation = self.quantize_activation) 
            self.top_l = self.create_mlp(ln_top, sigmoid_top, quant_linear_layer = self.quantize_act_and_lin, channelwise_lin = self.channelwise_lin, quantize_activation = self.quantize_activation) 
            if self.quantize_activation: 
                print("activation is quantized") 
            else: 
                print("not quantize activations, quantize weights") 

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                ) 

    def apply_mlp(self, x, layers, prev_act_scaling_factor = None): 
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers 
        '''
        count = 0 
        if not self.quantization_flag: # TODO recheck intentional reverse logic updated: check 
            return layers(x) 
        else: 
            for layer in layers: 
                if isinstance(layer, QuantLinear): 
                    x, prev_act_scaling_factor = layer(x, prev_act_scaling_factor) 
                    print("oooooooooooooooooo LAYER {} ooooooooooooooooo".format(count)) 
                    print("output") 
                    print(x) 
                    print("scale") 
                    print(prev_act_scaling_factor) 
                    count += 1 
                else: 
                    x = layer(x) 
            return x 
        ''' 
        count = 0 
        if args.quantization_flag: 
            for layer in layers.module: 
                if isinstance(layer, QuantLinear): 
                    global change_bitw, change_bitw2 
                    if change_bitw: 
                        self.weight_bit = change_bitw2 
                        layer.weight_bit = change_bitw2 
                        layer.bias_bit = change_bitw2 
                        print("change bit width to {}".format(change_bitw2)) 
                    
                    if self.change_lin_from_full_to_quantized: 
                        layer.full_precision_flag = False 
                        print("from full to {} bit quantized".format(layer.weight_bit)) 
                    '''
                    # identifying layer count 
                    print("layer", count) 
                    count += 1 
                    ''' 
                    x, prev_act_scaling_factor = layer(x, prev_act_scaling_factor) 
                    '''
                    print("layer {}".format(count)) 
                    print(prev_act_scaling_factor.shape) 
                    ''' 
                    '''
                    print("ooooooooooooooooooo LAYER {} oooooooooooooooooooo".format(count)) 
                    print("output") 
                    print(x[0 : 10]) 
                    print("scale") 
                    print(prev_act_scaling_factor) 
                    count += 1 
                elif isinstance(layer, nn.Linear): 
                    x = layer(x) 
                    print("ooooooooooooooooooo LAYER {} oooooooooooooooooooo".format(count)) 
                    print("output") 
                    print(x[0 : 10]) 
                    count += 1 
                ''' 
                else: 
                    x = layer(x) 
                    # print out the layer weights
        else: 
            x = layers(x) 
        return x 

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l, test_mode = False): 
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                print("per sample weights found") 
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = emb_l[k] 
                if self.quantization_flag: 
                    '''
                    V = E(
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights, 
                        full_precision_flag = full_precision_flag, 
                        test_mode = test_mode 
                    ) 
                    ''' 
                    V = E(
                        sparse_index_group_batch, 
                        sparse_offset_group_batch, 
                        per_sample_weights = per_sample_weights, 
                        full_precision_flag = True, 
                        test_mode = test_mode 
                    )
                else: 
                    V = E(
                        sparse_index_group_batch, 
                        sparse_offset_group_batch, 
                        per_sample_weights = per_sample_weights 
                    ) 

                ly.append(V)

        # print(ly)
        return ly

    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):

        n = len(self.emb_l)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):

        if not self.modify_feature_interaction: 
            # no quantization 
            if self.arch_interaction_op == "dot": 
                # concatenate dense and sparse features
                (batch_size, d) = x.shape
                T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
                # perform a dot product
                Z = torch.bmm(T, torch.transpose(T, 1, 2))
                # append dense feature with the interactions (into a row vector)
                # approach 1: all
                # Zflat = Z.view((batch_size, -1))
                # approach 2: unique
                _, ni, nj = Z.shape
                # approach 1: tril_indices
                # offset = 0 if self.arch_interaction_itself else -1
                # li, lj = torch.tril_indices(ni, nj, offset=offset)
                # approach 2: custom
                offset = 1 if self.arch_interaction_itself else 0
                li = torch.tensor([i for i in range(ni) for j in range(i + offset)]) 
                lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
                Zflat = Z[:, li, lj]
                # concatenate dense features and interactions
                R = torch.cat([x] + [Zflat], dim=1)
            elif self.arch_interaction_op == "cat":
                # concatenation features (into a row vector)
                R = torch.cat([x] + ly, dim=1)
            else:
                sys.exit(
                    "ERROR: --arch-interaction-op="
                    + self.arch_interaction_op
                    + " is not supported"
                ) 
            
            # change on return values 
            '''
            return R 
            ''' 
        else: 
            # quantization used to make possible bmm operation in context of integers 
            if not self.quantization_flag: 
                print("Warning: modify interaction features only happens when quantization_flag is set now it is not set results is not expected to be valid") 
            if self.arch_interaction_op == "dot": 
                (batch_size, d) = x.shape 
                T = torch.cat([x] + ly, dim = 1).view((batch_size, -1, d)) 
                x_min = T.data.min() 
                x_max = T.data.max() 
                '''
                # Initialization 
                if self.feature_xmin == self.feature_xmax: 
                    self.feature_xmin += x_min 
                    self.feature_xmax += x_max 
                
                else: 
                    self.feature_xmin = self.feature_xmin * 0.95 + x_min * (1 - 0.95) 
                    self.feature_xmax = self.feature_xmax * 0.95 + x_max * (1 - 0.95) 
                ''' 
                self.feature_xmin = x_min 
                self.feature_xmax = x_max 
            
                # finding scale 
                self.feature_scaling_factor = symmetric_linear_quantization_params(16, self.feature_xmin, self.feature_xmax, False) 

                T_integers = SymmetricQuantFunction.apply(T, 16, self.feature_scaling_factor) # TODO recheck activation_bit 

                Z_integers = torch.bmm(T_integers, torch.transpose(T_integers, 1, 2)) 

                Z = Z_integers * (self.feature_scaling_factor ** 2) 
                '''
                with torch.no_grad(): 
                    print("max bound") 
                    print(self.feature_xmax) 
                    print("min bound") 
                    print(self.feature_xmin) 
                    print("Z_integer * (feature_scaling_factor ** 2)") 
                    print(Z[0 : 10]) 
                    print("T dot production with T") 
                    print(torch.bmm(T, torch.transpose(T, 1, 2))[0 : 10]) 
                ''' 
                # incorporate features are copied 
                _, ni, nj = Z.shape
                # approach 1: tril_indices
                # offset = 0 if self.arch_interaction_itself else -1
                # li, lj = torch.tril_indices(ni, nj, offset=offset)
                # approach 2: custom
                offset = 1 if self.arch_interaction_itself else 0
                li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
                lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
                Zflat = Z[:, li, lj]
                # concatenate dense features and interactions
                R = torch.cat([x] + [Zflat], dim=1) 

                # since Z is of different scale 
                # quantize again 
            elif self.arch_interaction_op == "cat": 
                # concatenation is copied 
                R = torch.cat([x] + ly, dim=1) 
        if not self.quantization_flag:  # recheck intentional reverse logic updated: check 
            return R 
        else: 
            if self.quantize_activation: 
                R, feature_scaling_factor_at_bmlp = self.quant_feature_outputs(R) 
                return R, feature_scaling_factor_at_bmlp 
            else: 
                return R, None 


    def forward(self, dense_x, lS_o, lS_i, test_mode = False): 
        batch_size = dense_x.size()[0] 
        # if batch_size < ext_dist_two.my_size:
        if batch_size < ext_dist_three.my_size: 
            sys.exit(
                "ERROR: batch_size (%d) must be larger than number of ranks (%d)"
                % (batch_size, ext_dist_three.my_size) # batch_size, ext_dist_two.my_size 
            ) 
        # if batch_size % ext_dist_two.my_size != 0: 
        if batch_size % ext_dist_three.my_size != 0: 
            sys.exit(
                "ERROR: batch_size %d can not split across %d ranks evenly"
                % (batch_size, ext_dist_three.my_size) # batch_size, ext_dist_two.my_size
            ) 
        
        # dense_x = dense_x[ext_dist_two.get_my_slice(batch_size)] 
        dense_x = dense_x[ext_dist_three.get_my_slice(batch_size)] 
        lS_o = lS_o[self.local_emb_slice] 
        lS_i = lS_i[self.local_emb_slice] 

        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit(
                "ERROR: corrupted model input detected in distributed_forward call"
            )
        
        # check whether mlp is converted from full precision to weight_bit quantized bit width 
        global change_lin_full_quantize 
        change_lin_full_quantize = False 
        if change_lin_full_quantize: 
            change_lin_full_quantize = False # clear flag 
            self.quantize_act_and_lin = True 
            self.change_lin_from_full_to_quantized = True 

        if not self.quantization_flag: 
            # process dense features (using bottom mlp), resulting in a row vector
            # debug prints
            # print("intermediate")
            # print(x.detach().cpu().numpy())

            # process sparse features(using embeddings), resulting in a list of row vectors
            ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, test_mode = test_mode) 
            # print(len(ly)) 
            # for y in ly:
            #     print(y.detach().cpu().numpy())
            '''
            print("rank {} len of ly: {}".format(ext_dist_two.my_rank, len(ly))) 
            for y in ly: 
                print("rank {} y {}".format(ext_dist_two.my_rank, y.detach().cpu().shape)) 
            ''' 
            if len(self.emb_l) != len(ly):
                sys.exit("ERROR: corrupted intermediate result in distributed_forward call") 
            
            # print("rank {} per_rank_table_splits: {}".format(ext_dist_two.my_rank, self.n_emb_per_rank)) 
            # a2a_req = ext_dist_two.alltoall(ly, self.n_emb_per_rank) 
            a2a_req = ext_dist_three.alltoall(ly, self.n_emb_per_rank) 
    
            x = self.apply_mlp(dense_x, self.bot_l) 

            ly = a2a_req.wait() 
            ''' 
            print("rank {} length of ly {}".format(ext_dist_two.my_rank, len(ly))) 
            for y in ly: 
                print("rank {} reduced yy {}".format(ext_dist_two.my_rank, y.detach().cpu().shape)) 
                dist.barrier() 
            ''' 
            ly = list(ly) 
            # print("length embedding table num: {}".format(len(ly))) 
            # print("rank {} ly 0 length: {} ly 1 length: {} ly 2 length: {} ly 3 length: {}".format(ext_dist_two.my_rank, len(ly[0]), len(ly[1]), len(ly[2]), len(ly[3]))) 

            # interact features (dense and sparse)
            z = self.interact_features(x, ly)
            # print(z.detach().cpu().numpy())
            '''
            z = []
            for k in range(ndevices):
                zk = self.interact_features(x[k], ly[k]) 
                z.append(zk) 
            ''' 
            # obtain probability of a click (using top mlp)
            p = self.apply_mlp(z, self.top_l)

            # clamp output if needed
            if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
                z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
            else:
                z = p

            return z 
        else: 
            if (not self.quantize_act_and_lin) and self.quantize_activation: 
                # used for cases where embedding tables are quantized while mlp is in full precision 
                x, act_scaling_factor = self.quant_input(dense_x) 
                x = self.apply_mlp(x, self.bot_l) # not used with scale 
                ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, test_mode = test_mode) 
                z, feature_scaling_factor = self.interact_features(x, ly) 
                p = self.apply_mlp(z, self.top_l) # not used with scale 
            elif not self.quantize_activation: 
                # this part has been modified to be distributed with hybrid parallelism 
                ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, test_mode = test_mode) 
                if len(self.emb_l) != len(ly): 
                    sys.exit("ERROR: corrupted intermediate result in distributed_forward call") 
                # a2a_req = ext_dist_two.alltoall(ly, self.n_emb_per_rank) 
                a2a_req = ext_dist_three.alltoall(ly, self.n_emb_per_rank) 
                x = self.apply_mlp(dense_x, self.bot_l, prev_act_scaling_factor = None) 
                ly = a2a_req.wait() 
                ly = list(ly) 
                z, feature_scaling_factor = self.interact_features(x, ly) 
                p = self.apply_mlp(z, self.top_l, prev_act_scaling_factor = feature_scaling_factor) 
            else: 
                # used for cases where embedding tables and mlp are quantized 
                x, act_scaling_factor = self.quant_input(dense_x) 
                if act_scaling_factor is None: 
                    print("tuple is x") 
                '''
                else: 
                    print("oooooooooooooooooooo First oooooooooooooooooooo") 
                    print("activation bit") 
                    print(self.quant_input.activation_bit) 
                    print("output") 
                    print(x[0]) 
                    print("scale") 
                    print(act_scaling_factor) 
                ''' 

                x = self.apply_mlp(x, self.bot_l, prev_act_scaling_factor = act_scaling_factor) 
                ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, test_mode = test_mode) 
                z, feature_scaling_factor = self.interact_features(x, ly) 
                p = self.apply_mlp(z, self.top_l, prev_act_scaling_factor = feature_scaling_factor) 
            # copy clamp 
            if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
                z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold)) 
            else:
                z = p

            global change_bitw 
            if change_bitw: 
                change_bitw = False 
                print("find that bit width change enables, all linear layers are with updated bit width") 
            
            if self.change_lin_from_full_to_quantized: 
                self.change_lin_from_full_to_quantized = False 
                print("find that lin from full to quantized enabled, changes made") 

            return z 
    
    def investigate_ddpgradient(self, rank): 
        with torch.no_grad(): 
            if rank == 0: 
                print("investigating top layer weight gradients: ") 
            for names, param in self.top_l.module.named_parameters(): 
                if len(param.grad.shape) > 1: 
                    # print(param.grad.shape) 
                    print("rank {} name {} part of the grad {}".format(rank, names, param.grad[0][: 10])) 
                else: 
                    print("rank {} name {} part of the grad {}".format(rank, names, param.grad[: 10] if param.grad.shape[0] >= 10 else param.grad)) 
            ext_dist_three.barrier() 

            if rank == 0: 
                print("investigating bottom layer weight gradients: ") 
            for names, param in self.bot_l.module.named_parameters(): 
                if len(param.grad.shape) > 1: 
                    # print(param.grad.shape) 
                    print("rank {} name {} part of the grad {}".format(rank, names, param.grad[0][: 10])) 
                else: 
                    print("rank {} name {} part of the grad {}".format(rank, names, param.grad[: 10] if param.grad.shape[0] >= 10 else param.grad)) 
            ext_dist_three.barrier() 
            
            if rank == 0: 
                print("investigating embedding layer weight gradients: ") 
            for embidex, emb in enumerate(self.emb_l): 
                # print(emb.weight.grad.shape) 
                # print("rank {} embedding table index {} part of the grad {}".format(rank, embidex, emb.weight.grad[0][: 10])) 
                print("rank {} embedding table index {} is the gradient sparsecoo? {}".format(rank, embidex, emb.weight.grad.is_sparse)) 
            ext_dist_three.barrier() 

    
    def documenting_weights_tables(self, path, epoch_num, iter_num, emb_quantized = True): 
        table_nums = [3, 6] 
        with torch.no_grad(): 
            for table_num in table_nums: 
                file_name = "table" + str(table_num) + "epoch" + str(epoch_num) + "iter" + str(iter_num) + "_" 
                '''
                file_name = "table" + str(table_num) + "epoch" + str(epoch_num) 
                ''' 
                if emb_quantized: 
                    embedding_table = self.emb_l[table_num].embedding_bag 
                    eb_scaling_factor = self.emb_l[table_num].eb_scaling_factor 
                else: 
                    embedding_table = self.emb_l[table_num] 
                if emb_quantized: 
                    file_name += "quantized" 
                file_name += ".txt" 

                print("Start documenting table {} weights in file {}".format(table_num, file_name)) 

                file_path = path + "/" + file_name 
                file = open(file_path, "a") 

                weight_list = embedding_table.weight.data.detach() 
                if emb_quantized: 
                    zero_point = torch.tensor(0.).cuda() 
                    weight_list = linear_quantize(weight_list, eb_scaling_factor, zero_point) 
                    weight_list = weight_list * eb_scaling_factor 
                for i in range(weight_list.shape[0]): 
                    row = "" 
                    for j in range(weight_list.shape[1]): 
                        row += str(weight_list[i][j].item()) 
                        if j != weight_list.shape[1] - 1: 
                            row += ", " 
                    file.write(row) 
                    file.write("\n") 
                file.close() 
                print("Documented table {} weights in file {}".format(table_num, file_name)) 

                '''
                if emb_quantized and table_num == 6: 
                    file_name = "table" + str(table_num) + "epoch" + str(epoch_num) + "_" + "gradient" 
                    file_name += ".txt" 

                    print("Start documenting table {} gradient in {}".format(table_num, file_name)) 

                    file_path = path + "/" + file_name 
                    file = open(file_path, "a") 

                    list_o_gradients = embedding_table.weight.grad.data.detach() 
                    for i in range(list_o_gradients.shape[0]): 
                        row = "" 
                        for j in range(list_o_gradients.shape[1]): 
                            row += str(list_o_gradients[i][j].item()) 
                            if j != list_o_gradients.shape[1] - 1: 
                                row += ", " 
                        file.write(row) 
                        file.write("\n") 
                    file.close() 
                    print("Documented table {} gradients in file {}".format(table_num, file_name)) 
                ''' 
    
    def show_output_linear_layer_grad(self, start = False): 
        with torch.no_grad(): 
            if self.top_l[-2].weight.grad is not None: 
                print("device: {}".format(self.deviceid)) 
                '''
                print(self.top_l[-2].weight.grad[0][: 20]) 
                ''' 
                print(self.top_l[-2].weight[0][: 20]) 
            if start: 
                print(self.top_l[-2].weight[0][: 20]) 


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value 

def inference_distributed(
    rank, 
    args, 
    dlrm, 
    test_ld, 
    device, 
    use_gpu, 
    log_iter = -1, 
    nbatches = -1, 
    nbatches_test = -1, 
    writer = None 
): 
    test_accu = 0
    test_samp = 0 
    
    scores = [] 
    targets = [] 
    
    # the function operates on one gpu 
    num_batch = len(test_ld) 
    for i, testBatch in enumerate(test_ld): 
        if nbatches > 0 and i >= nbatches: 
            break 
        
        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch 
        ) 
        
        # if ext_dist_two.my_size > 1 and X_test.size(0) % ext_dist_two.my_size != 0: 
        if ext_dist_three.my_size > 1 and X_test.size(0) % ext_dist_three.my_size != 0: 
            print("Warning: Skipping the batch %d with size %d" % (i, X_test.size(0))) 
            continue 

        Z_test = dlrm_wrap(
            X_test, 
            lS_o_test, 
            lS_i_test, 
            use_gpu, 
            device, 
            ndevices = 1, 
            test_mode=True 
        ) 
        
        if Z_test.is_cuda: 
            torch.cuda.synchronize() 
        # (_, batch_split_lengths) = ext_dist_two.get_split_lengths(X_test.size(0)) 
        (_, batch_split_lengths) = ext_dist_three.get_split_lengths(X_test.size(0)) 
        # if ext_dist_two.my_size > 1: 
        if ext_dist_three.my_size > 1: 
            # Z_test = ext_dist_two.all_gather(Z_test, batch_split_lengths) 
            Z_test = ext_dist_three.all_gather(Z_test, batch_split_lengths) 

        S_test = Z_test.detach().cpu().numpy() 
        T_test = T_test.detach().cpu().numpy() 
        
        # calculating roc auc score 
        scores.append(S_test) 
        targets.append(T_test) 
        
        mbs_test = T_test.shape[0] 
        A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8)) 
        
        test_accu += A_test 
        test_samp += mbs_test 
        
        if rank == 0 and i % 200 == 0: 
            print("steps testing: {}".format(float(i)/num_batch), end = "\r") 
        
        ext_dist_three.barrier() 
    
    print("rank: {} test_accu: {}".format(rank, test_accu)) 
    print("get out") 
    '''
    print(test_accu.type) 
    print(test_samp.type) 
    total_accu = torch.tensor([0.0]) 
    total_samp = torch.tensor([0.0]) 
    for i in range(args.nodes * args.gpus): 
        if i == rank: 
            test_accu_hol = torch.tensor([test_accu], dtype = torch.float32, requires_grad = False, device = device) 
        else: 
            test_accu_hol = torch.zeros(1, dtype = torch.float32, requires_grad = False, device = device) 
        try: 
            dist.broadcast(test_accu_hol, src = rank) 
        except: 
            print("cannot broadcast") 
        print("receive from source: {}".format(i)) 
        if i == rank: 
            total_samp_hol = torch.tensor([test_samp], dtype = torch.float32, requires_grad = False, device = device) 
        else: 
            total_samp_hol = torch.zeros(1, dtype = torch.float32, requires_grad = False, device = device) 
        dist.broadcast(total_samp_hol, src = rank) 
        print("others receive from source: {}".format(i)) 
        total_accu += test_accu_hol.data 
        total_samp += total_samp_hol.data 
    
    test_accu = total_accu.item() 
    test_samp = total_samp.item() 
    ''' 
    
    print("{} has test check {} and sample count {}".format(rank, test_accu, test_samp)) 
    
    acc_test = test_accu / test_samp 
    '''
    writer.add_scalar("Test/Acc", acc_test, log_iter) 
    ''' 
    
    if rank == 0: 
        model_metrics_dict = {
            "nepochs": args.nepochs,
            "nbatches": nbatches,
            "nbatches_test": nbatches_test,
            "state_dict": dlrm.state_dict(),
            "test_acc": acc_test,
        } 
    
    global best_acc_test 
    is_best = acc_test > best_acc_test 
    if is_best: 
        best_acc_test = acc_test 
        
    scores = np.concatenate(scores, axis = 0) 
    targets = np.concatenate(targets, axis = 0) 
    roc_auc = sklearn.metrics.roc_auc_score(targets, scores) 
    
    global best_auc_test 
    best_auc_test = roc_auc if roc_auc > best_auc_test else best_auc_test 
    
    '''
    print(
        " accuracy {:3.3f} %, best {:3.3f} %".format(
            acc_test * 100, best_acc_test * 100 
        ), 
        flush = True, 
    ) 
    ''' 
    if rank == 0: 
        print(
            " accuracy {:3.3f} %, best {:3.3f} %, roc auc score {:.4f}, best {:.4f}".format(
                acc_test * 100, best_acc_test * 100, roc_auc, best_auc_test), 
            flush = True 
            )
        return model_metrics_dict, is_best 
    else: 
        return 


def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 0

    if args.mlperf_logging:
        scores = []
        targets = []

    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break

        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )

        # Skip the batch if batch size not multiple of total ranks
        if ext_dist_three.my_size > 1 and X_test.size(0) % ext_dist_three.my_size != 0:
            print("Warning: Skiping the batch %d with size %d" % (i, X_test.size(0)))
            continue

        # forward pass
        Z_test = dlrm_wrap(
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        ### gather the distributed results on each rank ###
        # For some reason it requires explicit sync before all_gather call if
        # tensor is on GPU memory
        if Z_test.is_cuda:
            torch.cuda.synchronize()
        (_, batch_split_lengths) = ext_dist_three.get_split_lengths(X_test.size(0))
        if ext_dist_three.my_size > 1:
            Z_test = ext_dist_three.all_gather(Z_test, batch_split_lengths)

        if args.mlperf_logging:
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            scores.append(S_test)
            targets.append(T_test)
        else:
            with record_function("DLRM accuracy compute"):
                # compute loss and accuracy
                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = T_test.detach().cpu().numpy()  # numpy array

                mbs_test = T_test.shape[0]  # = mini_batch_size except last
                A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

                test_accu += A_test
                test_samp += mbs_test

    if args.mlperf_logging:
        with record_function("DLRM mlperf sklearn metrics compute"):
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            metrics = {
                "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "ap": sklearn.metrics.average_precision_score,
                "roc_auc": sklearn.metrics.roc_auc_score,
                "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
            }

        validation_results = {}
        for metric_name, metric_function in metrics.items():
            validation_results[metric_name] = metric_function(targets, scores)
            writer.add_scalar(
                "mlperf-metrics-test/" + metric_name,
                validation_results[metric_name],
                log_iter,
            )
        acc_test = validation_results["accuracy"]
    else:
        acc_test = test_accu / test_samp
        writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    if args.mlperf_logging:
        is_best = validation_results["roc_auc"] > best_auc_test
        if is_best:
            best_auc_test = validation_results["roc_auc"]
            model_metrics_dict["test_auc"] = best_auc_test
        print(
            "recall {:.4f}, precision {:.4f},".format(
                validation_results["recall"],
                validation_results["precision"],
            )
            + " f1 {:.4f}, ap {:.4f},".format(
                validation_results["f1"], validation_results["ap"]
            )
            + " auc {:.4f}, best auc {:.4f},".format(
                validation_results["roc_auc"], best_auc_test
            )
            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                validation_results["accuracy"] * 100, best_acc_test * 100
            ),
            flush=True,
        )
    else:
        is_best = acc_test > best_acc_test
        if is_best:
            best_acc_test = acc_test
        print(
            " accuracy {:3.3f} %, best {:3.3f} %".format(
                acc_test * 100, best_acc_test * 100
            ),
            flush=True,
        )
    return model_metrics_dict, is_best


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="13-512-256-64-16")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="512-256-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # quantization options args 
    parser.add_argument("--quantization_flag", action = "store_true", default = False) 
    parser.add_argument("--embedding_bit", type = int, default = None) 
    parser.add_argument("--weight_bit", type = int, default = None) 
    # activations and loss 
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # mlperf gradient accumulation iterations
    parser.add_argument("--mlperf-grad-accum-iter", type=int, default=1)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    parser.add_argument("--investigating-inputs", action = "store_true", default = False) 
    parser.add_argument("--pretrain_and_quantize", action = "store_true", default = False) 
    parser.add_argument("--modify_feature_interaction", action = "store_true", default = False) 
    parser.add_argument("--linear_shift_down_bit_width", action = "store_true", default = False) 
    parser.add_argument("--documenting_table_weight", action = "store_true", default = False) 
    parser.add_argument("--pretrain_and_quantize_lin", action = "store_true", default = False) 
    parser.add_argument("--quantize_activation", action = "store_true", default = False) 
    parser.add_argument("--linear_channel", action = "store_true", default = False) 
    parser.add_argument("--quantize_act_and_lin", action = "store_true", default = False) 

    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()

    if args.dataset_multiprocessing:
        assert float(sys.version[:3]) > 3.7, "The dataset_multiprocessing " + \
        "flag is susceptible to a bug in Python 3.7 and under. " + \
        "https://github.com/facebookresearch/dlrm/issues/172"

    if args.mlperf_logging:
        mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)
        mlperf_logger.log_start(
            key=mlperf_logger.constants.INIT_START, log_all_ranks=True
        )

    if args.weighted_pooling is not None:
        if args.qr_flag:
            sys.exit("ERROR: quotient remainder with weighted pooling is not supported")
        if args.md_flag:
            sys.exit("ERROR: mixed dimensions with weighted pooling is not supported")
    if args.quantize_emb_with_bit in [4, 8]:
        if args.qr_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with quotient remainder is not supported"
            )
        if args.md_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with mixed dimensions is not supported"
            )
        if args.use_gpu:
            sys.exit(
                "ERROR: 4 and 8-bit quantization on GPU is not supported"
            )

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if not args.debug_mode:
        ext_dist_three.init_distributed(local_rank=args.local_rank, use_gpu=use_gpu, backend=args.dist_backend)

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        if ext_dist_three.my_size > 1:
            ngpus = 1
            device = torch.device("cuda", ext_dist_three.my_local_rank)
        else:
            ngpus = torch.cuda.device_count()
            device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    if args.mlperf_logging:
        mlperf_logger.barrier()
        mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
        mlperf_logger.barrier()
        mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
        mlperf_logger.barrier()

    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
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
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

    args.ln_emb = ln_emb.tolist()
    if args.mlperf_logging:
        print("command line args: ", json.dumps(vars(args)))

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

            torch.set_printoptions(precision=4)
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break
            print("mini-batch: %d" % j)
            print(X.detach().cpu())
            # transform offsets to lengths when printing
            print(
                torch.IntTensor(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                        ).tolist()
                        for i, S_o in enumerate(lS_o)
                    ]
                )
            )
            print([S_i.detach().cpu() for S_i in lS_i])
            print(T.detach().cpu())

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function
    )

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l, dlrm.v_W_l = dlrm.create_emb(
                m_spa, ln_emb, args.weighted_pooling
            )
        else:
            if dlrm.weighted_pooling == "fixed":
                for k, w in enumerate(dlrm.v_W_l):
                    dlrm.v_W_l[k] = w.cuda()

    # distribute data parallel mlps
    if ext_dist_three.my_size > 1:
        if use_gpu:
            device_ids = [ext_dist_three.my_local_rank]
            dlrm.bot_l = ext_dist_three.DDP(dlrm.bot_l, device_ids=device_ids)
            dlrm.top_l = ext_dist_three.DDP(dlrm.top_l, device_ids=device_ids)
        else:
            dlrm.bot_l = ext_dist_three.DDP(dlrm.bot_l)
            dlrm.top_l = ext_dist_three.DDP(dlrm.top_l)

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
            if ext_dist_three.my_size == 1
            else [
                {
                    "params": [p for emb in dlrm.emb_l for p in emb.parameters()],
                    "lr": args.learning_rate,
                },
                # TODO check this lr setup
                # bottom mlp has no data parallelism
                # need to check how do we deal with top mlp
                {
                    "params": dlrm.bot_l.parameters(),
                    "lr": args.learning_rate,
                },
                {
                    "params": dlrm.top_l.parameters(),
                    "lr": args.learning_rate,
                },
            ]
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )

    ### main loop ###

    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    if args.mlperf_logging:
        mlperf_logger.mlperf_submission_log("dlrm")
        mlperf_logger.log_event(
            key=mlperf_logger.constants.SEED, value=args.numpy_rand_seed
        )
        mlperf_logger.log_event(
            key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=args.mini_batch_size
        )

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device("cuda")
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        if args.mlperf_logging:
            ld_gAUC_test = ld_model["test_auc"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )
        if args.mlperf_logging:
            print(
                "Testing state: accuracy = {:3.3f} %, auc = {:.3f}".format(
                    ld_acc_test * 100, ld_gAUC_test
                )
            )
        else:
            print("Testing state: accuracy = {:3.3f} %".format(ld_acc_test * 100))

    if args.inference_only:
        # Currently only dynamic quantization with INT8 and FP16 weights are
        # supported for MLPs and INT4 and INT8 weights for EmbeddingBag
        # post-training quantization during the inference.
        # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)
        assert args.quantize_mlp_with_bit in [
            8,
            16,
            32,
        ], "only support 8/16/32-bit but got {}".format(args.quantize_mlp_with_bit)
        assert args.quantize_emb_with_bit in [
            4,
            8,
            32,
        ], "only support 4/8/32-bit but got {}".format(args.quantize_emb_with_bit)
        if args.quantize_mlp_with_bit != 32:
            if args.quantize_mlp_with_bit in [8]:
                quantize_dtype = torch.qint8
            else:
                quantize_dtype = torch.float16
            dlrm = torch.quantization.quantize_dynamic(
                dlrm, {torch.nn.Linear}, quantize_dtype
            )
        if args.quantize_emb_with_bit != 32:
            dlrm.quantize_embedding(args.quantize_emb_with_bit)
            # print(dlrm)

    print("time/loss/accuracy (if enabled):")

    if args.mlperf_logging:
        # LR is logged twice for now because of a compliance checker bug
        mlperf_logger.log_event(
            key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate
        )
        mlperf_logger.log_event(
            key=mlperf_logger.constants.OPT_LR_WARMUP_STEPS,
            value=args.lr_num_warmup_steps,
        )

        # use logging keys from the official HP table and not from the logging library
        mlperf_logger.log_event(
            key="sgd_opt_base_learning_rate", value=args.learning_rate
        )
        mlperf_logger.log_event(
            key="lr_decay_start_steps", value=args.lr_decay_start_step
        )
        mlperf_logger.log_event(
            key="sgd_opt_learning_rate_decay_steps", value=args.lr_num_decay_steps
        )
        mlperf_logger.log_event(key="sgd_opt_learning_rate_decay_poly_power", value=2)

    tb_file = "./" + args.tensor_board_filename
    writer = SummaryWriter(tb_file)

    ext_dist_three.barrier()
    with torch.autograd.profiler.profile(
        args.enable_profiling, use_cuda=use_gpu, record_shapes=True
    ) as prof:
        if not args.inference_only:
            k = 0
            total_time_begin = 0
            while k < args.nepochs:
                if args.mlperf_logging:
                    mlperf_logger.barrier()
                    mlperf_logger.log_start(
                        key=mlperf_logger.constants.BLOCK_START,
                        metadata={
                            mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1),
                            mlperf_logger.constants.EPOCH_COUNT: 1,
                        },
                    )
                    mlperf_logger.barrier()
                    mlperf_logger.log_start(
                        key=mlperf_logger.constants.EPOCH_START,
                        metadata={mlperf_logger.constants.EPOCH_NUM: (k + 1)},
                    )

                if k < skip_upto_epoch:
                    continue

                if args.mlperf_logging:
                    previous_iteration_time = None

                for j, inputBatch in enumerate(train_ld):
                    if j == 0 and args.save_onnx:
                        X_onnx, lS_o_onnx, lS_i_onnx, _, _, _ = unpack_batch(inputBatch)

                    if j < skip_upto_batch:
                        continue

                    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

                    if args.mlperf_logging:
                        current_time = time_wrap(use_gpu)
                        if previous_iteration_time:
                            iteration_time = current_time - previous_iteration_time
                        else:
                            iteration_time = 0
                        previous_iteration_time = current_time
                    else:
                        t1 = time_wrap(use_gpu)

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break

                    # Skip the batch if batch size not multiple of total ranks
                    if ext_dist_three.my_size > 1 and X.size(0) % ext_dist_three.my_size != 0:
                        print(
                            "Warning: Skiping the batch %d with size %d"
                            % (j, X.size(0))
                        )
                        continue

                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

                    # forward pass
                    Z = dlrm_wrap(
                        X,
                        lS_o,
                        lS_i,
                        use_gpu,
                        device,
                        ndevices=ndevices,
                    )

                    if ext_dist_three.my_size > 1:
                        T = T[ext_dist_three.get_my_slice(mbs)]
                        W = W[ext_dist_three.get_my_slice(mbs)]

                    # loss
                    E = loss_fn_wrap(Z, T, use_gpu, device)

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    # training accuracy is not disabled
                    # S = Z.detach().cpu().numpy()  # numpy array
                    # T = T.detach().cpu().numpy()  # numpy array

                    # # print("res: ", S)

                    # # print("j, train: BCE ", j, L)

                    # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    # A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                    with record_function("DLRM backward"):
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        if (args.mlperf_logging and (j + 1) % args.mlperf_grad_accum_iter == 0) or not args.mlperf_logging:
                            optimizer.zero_grad()
                        # backward pass
                        E.backward()

                        # optimizer
                        if (args.mlperf_logging and (j + 1) % args.mlperf_grad_accum_iter == 0) or not args.mlperf_logging:
                            optimizer.step()
                            lr_scheduler.step()

                    if args.mlperf_logging:
                        total_time += iteration_time
                    else:
                        t2 = time_wrap(use_gpu)
                        total_time += t2 - t1

                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                    )
                    should_test = (
                        (args.test_freq > 0)
                        and (args.data_generation in ["dataset", "random"])
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        train_loss = total_loss / total_samp
                        total_loss = 0

                        str_run_type = (
                            "inference" if args.inference_only else "training"
                        )

                        wall_time = ""
                        if args.print_wall_time:
                            wall_time = " ({})".format(time.strftime("%H:%M"))

                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                str_run_type, j + 1, nbatches, k, gT
                            )
                            + " loss {:.6f}".format(train_loss)
                            + wall_time,
                            flush=True,
                        )

                        log_iter = nbatches * k + j + 1
                        writer.add_scalar("Train/Loss", train_loss, log_iter)

                        total_iter = 0
                        total_samp = 0

                    # testing
                    if should_test:
                        epoch_num_float = (j + 1) / len(train_ld) + k + 1
                        if args.mlperf_logging:
                            mlperf_logger.barrier()
                            mlperf_logger.log_start(
                                key=mlperf_logger.constants.EVAL_START,
                                metadata={
                                    mlperf_logger.constants.EPOCH_NUM: epoch_num_float
                                },
                            )

                        # don't measure training iter time in a test iteration
                        if args.mlperf_logging:
                            previous_iteration_time = None
                        print(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                        ) 
                        '''
                        model_metrics_dict, is_best = inference(
                            args,
                            dlrm,
                            best_acc_test,
                            best_auc_test,
                            test_ld,
                            device,
                            use_gpu,
                            log_iter,
                        )
                        ''' 
                        model_metrics_dict, is_best = inference_distributed(
                            ext_dist_three.my_rank, 
                            args, 
                            dlrm, 
                            test_ld, 
                            device, 
                            use_gpu, 
                            log_iter, 
                            nbatches, 
                            nbatches_test, 
                            writer 
                        ) 
                        if (
                            is_best
                            and not (args.save_model == "")
                            and not args.inference_only
                            and ext_dist_three.my_rank == 0 
                        ): 
                            model_metrics_dict["epoch"] = k
                            model_metrics_dict["iter"] = j + 1
                            model_metrics_dict["train_loss"] = train_loss
                            model_metrics_dict["total_loss"] = total_loss
                            model_metrics_dict[
                                "opt_state_dict"
                            ] = optimizer.state_dict()
                            # added to make sure even if internet crashes during training, at least one checkpoint is successfully saved 
                            # save_addr = args.save_model + str(((j + 1)/10240)%2) 
                            save_addr = args.save_model.split(".")[0] + str(((j + 1)/10240)%2) + ".pt" 
                            '''
                            print("Saving model to {}".format(args.save_model))
                            torch.save(model_metrics_dict, args.save_model)
                            ''' 
                            print("Saving model to {}".format(save_addr)) 
                            torch.save(model_metrics_dict, save_addr) 

                        if args.mlperf_logging:
                            mlperf_logger.barrier()
                            mlperf_logger.log_end(
                                key=mlperf_logger.constants.EVAL_STOP,
                                metadata={
                                    mlperf_logger.constants.EPOCH_NUM: epoch_num_float
                                },
                            )

                        # Uncomment the line below to print out the total time with overhead
                        # print("Total test time for this group: {}" \
                        # .format(time_wrap(use_gpu) - accum_test_time_begin))

                        if (
                            args.mlperf_logging
                            and (args.mlperf_acc_threshold > 0)
                            and (best_acc_test > args.mlperf_acc_threshold)
                        ):
                            print(
                                "MLPerf testing accuracy threshold "
                                + str(args.mlperf_acc_threshold)
                                + " reached, stop training"
                            )
                            break

                        if (
                            args.mlperf_logging
                            and (args.mlperf_auc_threshold > 0)
                            and (best_auc_test > args.mlperf_auc_threshold)
                        ):
                            print(
                                "MLPerf testing auc threshold "
                                + str(args.mlperf_auc_threshold)
                                + " reached, stop training"
                            )
                            if args.mlperf_logging:
                                mlperf_logger.barrier()
                                mlperf_logger.log_end(
                                    key=mlperf_logger.constants.RUN_STOP,
                                    metadata={
                                        mlperf_logger.constants.STATUS: mlperf_logger.constants.SUCCESS
                                    },
                                )
                            break

                if args.mlperf_logging:
                    mlperf_logger.barrier()
                    mlperf_logger.log_end(
                        key=mlperf_logger.constants.EPOCH_STOP,
                        metadata={mlperf_logger.constants.EPOCH_NUM: (k + 1)},
                    )
                    mlperf_logger.barrier()
                    mlperf_logger.log_end(
                        key=mlperf_logger.constants.BLOCK_STOP,
                        metadata={mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1)},
                    )
                k += 1  # nepochs
            if args.mlperf_logging and best_auc_test <= args.mlperf_auc_threshold:
                mlperf_logger.barrier()
                mlperf_logger.log_end(
                    key=mlperf_logger.constants.RUN_STOP,
                    metadata={
                        mlperf_logger.constants.STATUS: mlperf_logger.constants.ABORTED
                    },
                )
        else:
            print("Testing for inference only") 
            '''
            inference(
                args,
                dlrm,
                best_acc_test,
                best_auc_test,
                test_ld,
                device,
                use_gpu,
            ) 
            ''' 
            inference_distributed(
                ext_dist_three.my_rank, 
                args, 
                dlrm, 
                test_ld, 
                device, 
                use_gpu, 
                log_iter, 
                nbatches = nbatches, 
                nbatches_test = nbatches_test, 
                writer = writer 
            ) 

    # profiling
    if args.enable_profiling:
        time_stamp = str(datetime.datetime.now()).replace(" ", "_")
        with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cpu_time_total"
                )
            )
        with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        """
        # workaround 1: tensor -> list
        if torch.is_tensor(lS_i_onnx):
            lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
        # workaound 2: list -> tensor
        lS_i_onnx = torch.stack(lS_i_onnx)
        """
        # debug prints
        # print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
        # print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))
        dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
        batch_size = X_onnx.shape[0]
        print("X_onnx.shape", X_onnx.shape)
        if torch.is_tensor(lS_o_onnx):
            print("lS_o_onnx.shape", lS_o_onnx.shape)
        else:
            for oo in lS_o_onnx:
                print("oo.shape", oo.shape)
        if torch.is_tensor(lS_i_onnx):
            print("lS_i_onnx.shape", lS_i_onnx.shape)
        else:
            for ii in lS_i_onnx:
                print("ii.shape", ii.shape)

        # name inputs and outputs
        o_inputs = (
            ["offsets"]
            if torch.is_tensor(lS_o_onnx)
            else ["offsets_" + str(i) for i in range(len(lS_o_onnx))]
        )
        i_inputs = (
            ["indices"]
            if torch.is_tensor(lS_i_onnx)
            else ["indices_" + str(i) for i in range(len(lS_i_onnx))]
        )
        all_inputs = ["dense_x"] + o_inputs + i_inputs
        # debug prints
        print("inputs", all_inputs)

        # create dynamic_axis dictionaries
        do_inputs = (
            [{"offsets": {1: "batch_size"}}]
            if torch.is_tensor(lS_o_onnx)
            else [
                {"offsets_" + str(i): {0: "batch_size"}} for i in range(len(lS_o_onnx))
            ]
        )
        di_inputs = (
            [{"indices": {1: "batch_size"}}]
            if torch.is_tensor(lS_i_onnx)
            else [
                {"indices_" + str(i): {0: "batch_size"}} for i in range(len(lS_i_onnx))
            ]
        )
        dynamic_axes = {"dense_x": {0: "batch_size"}, "pred": {0: "batch_size"}}
        for do in do_inputs:
            dynamic_axes.update(do)
        for di in di_inputs:
            dynamic_axes.update(di)
        # debug prints
        print(dynamic_axes)
        # export model
        torch.onnx.export(
            dlrm,
            (X_onnx, lS_o_onnx, lS_i_onnx),
            dlrm_pytorch_onnx_file,
            verbose=True,
            opset_version=11,
            input_names=all_inputs,
            output_names=["pred"],
            dynamic_axes=dynamic_axes,
        )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
    total_time_end = time_wrap(use_gpu)


if __name__ == "__main__":
    run()
