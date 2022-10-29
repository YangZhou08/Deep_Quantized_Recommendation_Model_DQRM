import argparse
import os
import random
import shutil
import time
import logging
import warnings

import builtins 
import datetime 
import json 
import sys 
import time 

import dlrm_data_pytorch as dp 

import extend_distributed as ext_dist 

# mixed dimension 
from torch.utils.tensorboard import SummaryWriter 

# quotient-reminder 
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim 
from torch.optim.lr_scheduler import _LRScheduler 
from torch.nn.parameter import Parameter 
import sklearn.metrics 

from torch._ops import ops 
from torch.autograd.profiler import record_function 

from quantization_supp.quant_modules import QuantEmbeddingBagTwo 
from quantization_supp.quant_modules import QuantLinear 
from quantization_supp.quant_modules import QuantAct 
from quantization_supp.quant_utils import symmetric_linear_quantization_params 
from quantization_supp.quant_utils import SymmetricQuantFunction 
from quantization_supp.full_precision_modules import EmbeddingBagCompressedGrad 
from quantization_supp.full_precision_modules import LinearCompressedGrad 

from sgd_quantized_gradients import clear_gradients 
from sgd_quantized_gradients import weights_update 
from sgd_quantized_gradients import weights_update_added_quantization 
from sgd_quantized_gradients import grad_buffer_zeroing 
from sgd_quantized_gradients import grad_buffer_update 
from sgd_quantized_gradients import grad_buffer_update_added_quantization 

import optim.rwsadagrad as RowWiseSparseAdagrad 

'''
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print("Unable to import onnx. ", error)
''' 

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter 

best_acc_test = 0 
best_auc_test = 0 
full_precision_flag = True 
path_log = None 
iteration_num = 0 
change_bitw = False 
change_bitw2 = 4 
dlrm = None 
args = None 
buffer_clean = False 

change_lin_full_quantize = False 

exc = getattr(builtins, "IOError", "FileNotFoundError")


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, ndevices=1, test_mode = False): 
    '''
    print("print inside dlrm_wrap ndevices is ", ndevices) 
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
        return dlrm(X.to(device), lS_o, lS_i, test_mode = test_mode) 


def loss_fn_wrap(Z, T, use_gpu, device, args): 
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce": 
            '''
            return dlrm.loss_fn(Z, T.to(device)) 
            ''' 
            if args.loss_function == "mse": 
                loss_fn = torch.nn.MSELoss(reduction="mean") 
            else: 
                loss_fn = torch.nn.BCELoss(reduction="mean") 
            return loss_fn(Z, T.to(device)) 
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device) 
            loss_fn = torch.nn.BCELoss(reduction="none") 
            '''
            loss_fn_ = dlrm.loss_fn(Z, T.to(device)) 
            ''' 
            loss_fn_ = loss_fn(Z, T.to(device)) 
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
                print("using layer linear with compressed gradient (optional)") 
                lnr = LinearCompressedGrad(gradient_precision = 32) 
                lnr.set_param(LL) 
                layers.append(lnr) 
            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid()) 
            else:
                layers.append(nn.ReLU()) 

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers 
        '''
        if not self.quantization_flag: # TODO recheck intentionally reversed logic updated: checked 
            return torch.nn.Sequential(*layers) 
        else: 
            return layers 
        ''' 
        return layers 

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = [] 
        print("world size found is " + str(ext_dist.my_size)) 
        for i in range(0, ln.size): 
            # each gpu will have all embedding tables 
            '''
            if ext_dist.my_size > 1:
                if i not in self.local_emb_indices:
                    continue 
            ''' 
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
                '''
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True) 
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n)) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32) 
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True) 
                ''' 
                print("---------- Embedding Table {}, quantization not used, n = {}, m = {}, gradient compression precision set to {}".format(i, n, m, 8)) 
                EE = EmbeddingBagCompressedGrad(n, m) 
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
        deviceid = None 
    ): 
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

            if self.quantization_flag: 
                if self.weight_bit is not None: 
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
            if ext_dist.my_size > 1:
                n_emb = len(ln_emb)
                if n_emb < ext_dist.my_size:
                    sys.exit(
                        "only (%d) sparse features for (%d) devices, table partitions will fail"
                        % (n_emb, ext_dist.my_size)
                    )
                self.n_global_emb = n_emb
                self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(
                    n_emb
                )
                self.local_emb_slice = ext_dist.get_my_slice(n_emb)
                self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

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
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot) 
            ''' 
            if self.quantize_act_and_lin: 
                self.bot_l = self.create_mlp(ln_bot, sigmoid_bot, quant_linear_layer = True, channelwise_lin = self.channelwise_lin, quantize_activation = self.quantize_activation) 
                self.top_l = self.create_mlp(ln_top, sigmoid_top, quant_linear_layer = True, channelwise_lin = self.channelwise_lin, quantize_activation = self.quantize_activation) 
            else: 
                self.bot_l = self.create_mlp(ln_bot, sigmoid_bot) 
                self.top_l = self.create_mlp(ln_top, sigmoid_top) 
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
        for layer in layers: 
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
                    V = E(
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights, 
                        full_precision_flag = full_precision_flag, 
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
        # check whether mlp is converted from full precision to weight_bit quantized bit width 
        global change_lin_full_quantize 
        if change_lin_full_quantize: 
            change_lin_full_quantize = False # clear flag 
            self.quantize_act_and_lin = True 
            self.change_lin_from_full_to_quantized = True 

        if not self.quantization_flag: 
            # process dense features (using bottom mlp), resulting in a row vector
            x = self.apply_mlp(dense_x, self.bot_l)
            # debug prints
            # print("intermediate")
            # print(x.detach().cpu().numpy())

            # process sparse features(using embeddings), resulting in a list of row vectors
            ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, test_mode = test_mode) 
            # for y in ly:
            #     print(y.detach().cpu().numpy())

            # interact features (dense and sparse)
            z = self.interact_features(x, ly)
            # print(z.detach().cpu().numpy())

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
                # used when activations are not quantized 
                x = self.apply_mlp(dense_x, self.bot_l, prev_act_scaling_factor = None) 
                ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, test_mode = test_mode) 
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
    
    def documenting_weights_tables(self, path, epoch_num): 
        with torch.no_grad(): 
            for j, embedding_table in enumerate(self.emb_l): 
                file_name = "table" + str(j) + "epoch" + str(epoch_num) + ".txt" 
                file_path = path + "/" + file_name 
                file = open(file_path, "a") 
                weight_list = embedding_table.weight.data.detach() 
                for i in range(weight_list.shape[0]): 
                    row = "" 
                    for j in range(weight_list.shape[1]): 
                        row += str(weight_list[i][j].item()) 
                        if j != weight_list.shape[1] - 1: 
                            row += ", " 
                    file.write(row) 
                    file.write("\n") 
                file.close() 
    
    def show_output_linear_layer_grad(self): 
        with torch.no_grad(): 
            if self.top_l[-2].weight.grad is not None: 
                print("device: {}".format(self.deviceid)) 
                print("showing gradients") 
                print(self.top_l[-2].weight.grad[0][: 20]) 
                print("showing buffer") 
                print(self.top_l[-2].weight_grad_buffer[0][: 20]) 

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

def main(): 
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
    parser.add_argument("--number_of_cpu_node", type = int, default = 1) 

    global args
    global nbatches
    global nbatches_test
    global writer 
    global change_bitw 
    global change_bitw2 

    args = parser.parse_args() 
    args.world_size = 1 
    
    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers 
    if args.pretrain_and_quantize_lin: 
        args.quantize_act_and_lin = False 
    if args.linear_channel: 
        args.quantize_activation = False 

    train(0, args) 

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
        
        if X_test.size(0) % args.world_size != 0: 
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
        
        S_test = Z_test.detach().cpu().numpy() 
        T_test = T_test.detach().cpu().numpy() 
        
        # calculating roc auc score 
        scores.append(S_test) 
        targets.append(T_test) 
        
        mbs_test = T_test.shape[0] 
        A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8)) 
        
        test_accu += A_test 
        test_samp += mbs_test 
        '''
        if rank == 0 and i % 200 == 0: 
            print("steps testing: {}".format(float(i)/num_batch), end = "\r") 
        ''' 
        if i % 200 == 0: 
            print("steps testing: {}".format(float(i)/num_batch), end = "\r") 
    
    print("test_accu: {}".format(rank, test_accu)) 
    print("get out") 

    print("{} has test check {} and sample count {}".format(0, test_accu, test_samp)) 

    acc_test = test_accu / test_samp 

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

    print(
        " accuracy {:3.3f} %, best {:3.3f} %, roc auc score {:.4f}, best {:.4f}".format(
            acc_test * 100, best_acc_test * 100, roc_auc, best_auc_test), 
        flush = True 
        )
    return model_metrics_dict, is_best 

def train(gpu, args): 
    rank = gpu 
    print("number of gpu simulated is: {}".format(args.number_of_cpu_node)) 

    torch.manual_seed(0) 
    batch_size = args.mini_batch_size 

    torch.set_printoptions(profile = "full") 
    global buffer_clean 
    buffer_clean = False 

    global full_precision_flag 
    full_precision_flag = args.pretrain_and_quantize 

    global change_bitw 
    global change_bitw2 
    change_bitw = False 
    change_bitw2 = 8 

    global change_lin_full_quantize 
    change_lin_full_quantize = False 

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
    '''
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = args.world_size, rank = rank) 
    ''' 
    '''
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas = args.world_size, rank = rank) 
    ''' 
    
    collate_wrapper_criteo = dp.collate_wrapper_criteo_offset 
    
    train_loader = torch.utils.data.DataLoader( 
        dataset = train_dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = 0, 
        pin_memory = True, 
        collate_fn = collate_wrapper_criteo, 
        drop_last = False 
    ) 

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

    m_spa = args.arch_sparse_feature_size 
    ln_emb = np.asarray(ln_emb) 
    num_fea = ln_emb.size + 1 
    
    m_den_out = ln_bot[ln_bot.size - 1] 
    if args.arch_interaction_op == "dot": 
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
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype = int, sep = "-") 

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
        # for j, inputBatch in enumerate(train_ld):
        for j, inputBatch in enumerate(train_loader): 
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
    print("number of devices " + str(ndevices)) 
    
    global dlrm 
    dlrm = DLRM_Net(m_spa,
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
        loss_function=args.loss_function, 
        quantization_flag = args.quantization_flag, 
        embedding_bit = args.embedding_bit, 
        modify_feature_interaction = args.modify_feature_interaction, 
        weight_bit = 8 if args.linear_shift_down_bit_width else args.weight_bit, 
        quantize_act_and_lin = args.quantize_act_and_lin, 
        mlp_channelwise = args.linear_channel, 
        quantize_activation = args.quantize_activation, 
        deviceid = gpu) 

    global path_log 
    lstr = args.raw_data_file.split("/") 
    path_log = "/".join(lstr[0: -1]) + "/" 
    print("log path is written: {}".format(path_log)) 
    '''
    if rank == 0 and args.documenting_table_weight: 
        # record embedding table weight the first time 
        dlrm.documenting_weights_tables(path_log, 0) 
    dist.barrier() 
    ''' 
    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)
    
    '''
    dlrm.cuda(gpu) # TODO think about using cpu and change code 
    ''' 
    dlrm.to(device) 
    # TODO check whether the following section is supported 
    
    if dlrm.weighted_pooling == "fixed":
        for k, w in enumerate(dlrm.v_W_l):
            dlrm.v_W_l[k] = w.cuda() 
    
    '''
    dlrm.emb_l, dlrm.v_W_l = dlrm.create_emb(
        m_spa, ln_emb, args.weighted_pooling 
    ) 
    ''' 
    '''
    dlrm = nn.parallel.DistributedDataParallel(dlrm, device_ids = [gpu]) 
    ''' 
    if not args.inference_only: 
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]: # TODO check whether PyTorch support adagrad 
            sys.exit("GPU version of Adagrad is not supported by PyTorch.") 
        opts = {
            "sgd": torch.optim.SGD, 
            "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad, 
            "adagrad": torch.optim.Adagrad 
        } 
        
        parameters = (dlrm.parameters()) 
        print("optimizer selected is ", args.optimizer) 
        '''
        optimizer = opts[args.optimizer](parameters, lr = args.learning_rate) 
        ''' 
        '''
        optimizer = quantized_sgd(parameters, lr = args.learning_rate) 
        ''' 
        optimizer = opts[args.optimizer](parameters, lr = args.learning_rate) 
        lr_scheduler = LRPolicyScheduler(
            optimizer, 
            args.lr_num_warmup_steps, 
            args.lr_decay_start_step, 
            args.lr_num_decay_steps, 
        ) 
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0
    
    global best_acc_test 
    global best_auc_test 
    best_acc_test = 0
    best_auc_test = 0 
    
    if not (args.load_model == ""): 
        print("Loading saved model {}".format(args.load_model)) 
        # loading model can only be used for inference here 
        if use_gpu: 
            ld_model = torch.load(args.load_model, map_location = torch.device("cuda")) 
        else: 
            ld_model = torch.load(args.load_model, map_location = torch.device("cpu")) 
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
    
    tb_file = "./" + args.tensor_board_filename 
    writer = SummaryWriter(tb_file) 
    
    if args.investigating_inputs and gpu == 0: 
        print("investigating the training and testing input") 
        
        file = open("/rscratch/data/dlrm_criteo/investigating_input/investigating_input.txt", "a") 
        file.write("training set") 
        train_irregular_count = 0 
        test_irregular_count = 0 
        for j, inputBatch in enumerate(train_loader): 
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch) 
            '''
            file.write("Shape of X: {}\n".format(X.shape)) 
            ''' 
            if j % 1024 == 0: 
                file.write(str(lS_o)) 
                file.write(str(lS_i)) 
            if lS_o.shape[0] != 26 or lS_i.shape[0] != 26: 
                file.write("Batch: {}\n".format(j)) 
                file.write("Shape of lS_o: {}\n".format(lS_o.shape)) 
                file.write("Shape of lS_i: {}\n".format(lS_i.shape)) 
                train_irregular_count += 1 
        file.write("testing set") 
        for j, inputBatch in enumerate(test_loader): 
            '''
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch) 
            file.write("Batch: {}\n".format(j)) 
            file.write("Shape of X: {}\n".format(X.shape)) 
            file.write("Shape of lS_o: {}\n".format(lS_o.shape)) 
            file.write("Shape of lS_i: {}\n".format(lS_i.shape)) 
            ''' 
            if j % 1024 == 0: 
                file.write(str(lS_o)) 
                file.write(str(lS_i)) 
            if lS_o.shape[0] != 26 or lS_i.shape[0] != 26: 
                file.write("Batch: {}\n".format(j)) 
                file.write("Shape of lS_o: {}\n".format(lS_o.shape)) 
                file.write("Shape of lS_i: {}\n".format(lS_i.shape)) 
                test_irregular_count += 1 
        print("investigation completing") 
        print("encounter training dataset wanted inputs: {} encounter testing dataset wanted inputs: {}".format(train_irregular_count, test_irregular_count)) 
        print(train_irregular_count) 
        print(test_irregular_count) 
        return 
    
    # TODO use barrier if not in synchronization 
    
    if not args.inference_only: 
        k = 0 
        while k < args.nepochs: 
            if k == 1 and args.pretrain_and_quantize: 
                # one epoch of pretraining and one epoch of quantization-aware training 
                full_precision_flag = False 
                print("Using {}-bit precision".format(int(args.embedding_bit)) if args.embedding_bit is not None else "Still using full precision") 
            if args.pretrain_and_quantize_lin: 
                if k == 2: 
                    change_lin_full_quantize = True 
            
            if args.linear_shift_down_bit_width: 
                if k == 3: 
                    change_bitw = True 
                    change_bitw2 = args.weight_bit 
            if k < skip_upto_epoch: 
                continue 
            for j, inputBatch in enumerate(train_loader): 
                global iteration_num 
                iteration_num = j 

                # testing full lin to quantized 
                if j < skip_upto_batch: 
                    continue 
                
                X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch) 
                
                t1 = time_wrap(use_gpu) 
                
                if nbatches > 0 and j >= nbatches: 
                    break 
                
                if X.size(0) % args.world_size != 0: 
                    print(
                        "Warning: Skipping the batch %d with size %d" 
                        % (j, X.size(0)) 
                    )
                    continue 
                
                mbs = T.shape[0] 
                
                Z = dlrm_wrap(
                    X, 
                    lS_o, 
                    lS_i, 
                    use_gpu, 
                    device, 
                    ndevices = 1 # TODO check if ndevices is needed here 
                ) 
                
                # loss 
                # TODO check whether loss function can propagate through 
                E = loss_fn_wrap(Z, T, use_gpu, device, args) 
                
                L = E.detach().cpu().numpy() 
                
                # backward propagation 
                # tried to see if the gradients can be modified 
                '''
                optimizer.zero_grad() 
                ''' 
                clear_gradients(dlrm) # gradients zeroing (clearing) 

                if buffer_clean: 
                    '''
                    print("zeroing buffers") 
                    ''' 
                    grad_buffer_zeroing(dlrm) # buffers zeroing (clearing) 
                    buffer_clean = False 
                
                E.backward() 
                # quantization of gradient 
                '''
                print("updating buffers") 
                ''' 
                '''
                grad_buffer_update(dlrm, args.number_of_gpus) # update buffers 
                ''' 
                grad_buffer_update_added_quantization(dlrm, args.number_of_cpu_node, emb_grad_quantized = False) 

                if j % args.number_of_cpu_node == 0: 
                    '''
                    print("updating weights") 
                    ''' 
                    '''
                    weights_update(dlrm, lr_scheduler.get_lr()[-1]) 
                    ''' 
                    '''
                    if k >= 3: 
                        weights_update_added_quantization(dlrm, lr_scheduler.get_lr()[-1], args.number_of_gpus, emb_grad_quantized = False, update_embedding = False) 
                    else: 
                        weights_update_added_quantization(dlrm, lr_scheduler.get_lr()[-1], args.number_of_gpus, emb_grad_quantized = False) 
                    ''' 
                    weights_update_added_quantization(dlrm, lr_scheduler.get_lr()[-1], args.number_of_cpu_node, emb_grad_quantized = False, update_embedding = True) 
                    lr_scheduler.step() 
                    buffer_clean = True 
                '''
                dlrm.show_output_linear_layer_grad() 
                ''' 
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
                
                if should_test: 
                    # test on the first gpu on the first node 
                    epoch_num_float = (j + 1) / len(train_loader) 
                    print(
                        "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                    ) 
                    '''
                    if rank == 0: 
                    ''' 
                    model_metrics_dict, is_best = inference_distributed(
                        rank, 
                        args, 
                        dlrm, 
                        test_loader, 
                        device, 
                        use_gpu, 
                        log_iter, 
                        nbatches, 
                        nbatches_test, 
                        writer 
                    ) 
                    if not (args.save_model == "") and not args.inference_only: 
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
            k += 1 
    else: 
        print("Testing for inference only") 
        inference_distributed(
            rank, 
            args, 
            dlrm, 
            test_loader, 
            device, 
            use_gpu, 
            log_iter = -1, 
            nbatches = nbatches, 
            nbatches_test = nbatches_test, 
            writer = writer 
        ) 

        print("finish execution of inference") 
        if args.documenting_table_weight: 
            dlrm.module.documenting_weights_tables(path_log, 1) 
        return 
    
    if args.nr == 0 and gpu == 0: 
        if args.plot_compute_graph: 
            sys.exit(
                "ERROR: Please install pytorchviz package in order to use the"
                + " visualization. Then, uncomment its import above as well as"
                + " three lines below and run the code again."
            ) 

    if not args.inference_only: 
        # test on the first gpu on the first node 
        epoch_num_float = (j + 1) / len(train_loader) 
        print(
            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
        ) 
        model_metrics_dict, is_best = inference_distributed(
            rank, 
            args, 
            dlrm, 
            test_loader, 
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
            '''
            torch.save(model_metrics_dict, save_addr) 
            ''' 

if __name__ == "__main__": 
    main() 
