import torch
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable


def quantize_k(r_i, k):
	scale = (2**k - 1)
	r_o = torch.round( scale * r_i ) / scale
	return r_o

class DoReFaQuant(Function):
	@staticmethod
	def forward(ctx, r_i, k):
		tanh = torch.tanh(r_i).float()
		# scale = 2**k - 1.
		# quantize_k = torch.round( scale * (tanh / 2*torch.abs(tanh).max() + 0.5 ) ) / scale
		r_o = 2*quantize_k( tanh / (2*torch.max(torch.abs(tanh)).detach()) + 0.5 , k) - 1
		# r_o = 2 * quantize_k - 1.
		return r_o

	@staticmethod
	def backward(ctx, dLdr_o):
		# due to STE, dr_o / d_r_i = 1 according to formula (5)
		return dLdr_o, None


class QuantConv2dPACT(nn.Conv2d):
	def __init__(self, in_places, out_planes, kernel_size, stride=1, padding = 0, groups=1, dilation=1, bias = False, bitwidth = 8):
		super(QuantConv2dPACT, self).__init__(in_places, out_planes, kernel_size, stride, padding, groups, dilation, bias)
		self.quantize = DoReFaQuant.apply
		self.bitwidth = bitwidth

	def forward(self, x):
		vhat = self.quantize(self.weight, self.bitwidth)
		y = F.conv2d(x, vhat, self.bias, self.stride, self.padding, self.dilation, self.groups)
		return y

class QuantLinearPACT(nn.Linear):
	def __init__(self, in_features, out_features, bias = True, bitwidth = 8):
		super(QuantLinearPACT, self).__init__(in_features, out_features, bias)
		self.quantize = DoReFaQuant.apply
		self.bitwidth = bitwidth
	def forward(self, x):
		vhat = self.quantize(self.weight, self.bitwidth) # quantize the weights 
		if self.bias is not None: 
			vhatbias = self.quantize(self.bias, self.bitwidth) # quantize the bias
			return F.linear(x, weight = vhat, bias = vhatbias)
		else:
			return F.linear(x, weigth = vhat) 

# The following block of code is adopted from the course project for the UT Austin SysML class. 
# Specifically, I acknowledge the authorship of the following block of code to Shixing Yu (https://billysx.github.io/) 
class QuantEmbeddingBagPACT(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 bitwidth=8,
                 full_precision_flag=False,
                 embedding_id=None):
        super(QuantEmbeddingBagPACT, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bitwidth = bitwidth
        self.full_precision_flag = full_precision_flag
        self.embedding_id = embedding_id

        W = np.random.uniform(
            low=-np.sqrt(1 / self.num_embeddings),
            high=np.sqrt(1 / self.num_embeddings),
            size=(self.num_embeddings, self.embedding_dim)
        ).astype(np.float32)

        self.embedding_bag = nn.EmbeddingBag(
            self.num_embeddings, self.embedding_dim, mode="sum", sparse=True)
        self.embedding_bag.weight.data = torch.tensor(W, requires_grad=True)

        self.quantize = DoReFaQuant.apply

    def __repr__(self):
        # called when print(layer) 
        s = super(QuantEmbeddingBagPACT, self).__repr__()
        s = "(" + s + " embedding_bit = {}, full_precision_flag = {}, quant_mode = {})".format(
            self.embedding_bit, self.full_precision_flag, self.quant_mode
        )

    def set_params(self, embedding_bag):
        self.weight = Parameter(embedding_bag.weight.data.clone())

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False


    def forward(self, input, offsets=None, per_sample_weights=None, full_precision_flag = False, test_mode = False):
        if not full_precision_flag:
            vhat = self.quantize(self.embedding_bag.weight, self.bitwidth)
        else:
            vhat = self.embedding_bag.weight
        output = F.embedding_bag(input, weight = vhat, offsets = offsets, sparse = True, mode = "sum") # right now,
        '''
        scale_tanh = finding_tanh(self.embedding_bag.weight.data.clone()) 
        output = self.embedding_bag(input, offsets, per_sample_weights = None) 
        output = self.quantize(output, self.bitwidth, scale_tanh) 
        ''' 
        # output = self.embedding_bag(input, offsets, per_sample_weights, vhat)
        return output
