import math
import numpy as np
import torch
import time
import bisect
import decimal
from fractions import Fraction
from decimal import Decimal
from torch.autograd import Function, Variable

iteration_num = 0 


def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def transfer_conv_size(input_tensor):
    return input_tensor.view(1, -1, 1, 1)


def transfer_fc_size(input_tensor):
    return input_tensor.view(1, -1)


def transfer_numpy_float(inputs):
    inputs = inputs.reshape(-1)
    tmp_output = []

    for inp in inputs:
        tmp_output.append(float(inp))
    return np.array(tmp_output)


def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    """
    Calculate the percentile max and min values in a given tensor

    Parameters:
    ----------
    input: tensor
        the tensor to calculate percentile max and min
    lower_percentile: float
        if 0.1, means we return the value of the smallest 0.1% value in the tensor as percentile min
    upper_percentile: float
        if 99.9, means we return the value of the largest 0.1% value in the tensor as percentile max
    output_tensor: bool, default False
        if True, this function returns tensors, otherwise it returns values
    """
    input_length = input.shape[0]

    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)

    upper_bound = torch.kthvalue(input, k=upper_index).values

    if lower_percentile == 0:
        lower_bound = upper_bound * 0
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize floating point input tensor to integers with the given scaling factor and zeropoint.

    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # reshape scale and zeropoint for convolutional weights and activations 
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        if len(scale.shape) != 1 or scale.shape[0] != 1: 
            scale = scale.view(-1, 1) # ask whether this is valid TODO ask about the change 
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    if inplace:
        input.mul_(1. / scale.item()).to(torch.int32) 
        return input
    else: 
        return torch.round(1. / scale * input + zero_point) 

def linear_dequantize(input_q, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed-point floating point with given scaling factor and zeropoint.

    Parameters:
    ----------
    input_q: quantized integer tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # reshape scale and zeropoint for convolutional weights and activations
    if len(input_q.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input_q.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    # mapping integer input_q to fixed-point floating point value with given scaling factor and zeropoint
    if inplace:
        input_q.sub_(zero_point).mul_(scale)
        return input_q
    return (input_q - zero_point) * (scale) 

def finding_range_for_gradient(embedding_bag): 
    with torch.no_grad(): 
        if isinstance(embedding_bag, torch.nn.Module): 
            weight = embedding_bag.weight.data 
        else: 
            weight = embedding_bag 
        w_min, _ = torch.min(torch.min(weight, dim = 0, out = None).values, dim = 0, out = None) # no copy of the entire table is produced or we expected 
        w_max, _ = torch.max(torch.max(weight, dim = 0, out = None).values, dim = 0, out = None) # no copy of the entire table is produced or we expected 
        scale = max(w_min.abs(), w_max.abs()) 
        return scale 

def symmetric_linear_quantization_param_two(num_bits, 
                                            embedding_bag, 
                                            embedding_bound, 
                                            num_embeddings, 
                                            embedding_id): 
    # this function computes scale for embedding table only 
    # use with caution 
    
    with torch.no_grad(): 
        '''
        if num_embeddings > 1e6: 
            # using embedding table weight min and max 
            if isinstance(embedding_bag, torch.nn.Module): 
                weight = embedding_bag.weight.data 
            else: 
                weight = embedding_bag 
            w_min, _ = torch.min(torch.min(weight, dim = 0, out = None).values, dim = 0, out = None) # no copy of the entire table is produced or we expected 
            w_max, _ = torch.max(torch.max(weight, dim = 0, out = None).values, dim = 0, out = None) # no copy of the entire table is produced or we expected 

            if dlrm_s_pytorch_dp_only.iteration_num % 10240 == 0 and (embedding_id == 2 or embedding_id == 3): 
                print(w_min, w_max) 
            n = 2 ** (num_bits - 1) - 1 
        
            scale = max(w_min.abs(), w_max.abs()) 
            scale = torch.clamp(scale, min = 1e-8) / n 
        else: 
            # using initialization values 
            n = 2 ** (num_bits - 1) - 1 
            scale = embedding_bound 
            scale = torch.clamp(scale, min = 1e-8) / n 
        '''
        # using embedding table weight min and max 
        if isinstance(embedding_bag, torch.nn.Module): 
            weight = embedding_bag.weight.data 
        else: 
            weight = embedding_bag 
        w_min, _ = torch.min(torch.min(weight, dim = 0, out = None).values, dim = 0, out = None) # no copy of the entire table is produced or we expected 
        w_max, _ = torch.max(torch.max(weight, dim = 0, out = None).values, dim = 0, out = None) # no copy of the entire table is produced or we expected 
        '''
        global iteration_num 
        iteration_num += 1 

        if iteration_num % (26 * 200) == 3 or iteration_num % (26 * 200) == 4 or iteration_num % (26 * 200) == 21: 
            print("Note: Table {}, wmin {}, wmax {}".format(embedding_id, w_min, w_max)) 

        if iteration_num % (26 * 200) == 18 or iteration_num % (26 * 200) == 17 or iteration_num % (26 * 200) == 7: 
            print("Note: Table {}, wmin {}, wmax {}".format(embedding_id, w_min, w_max)) 
        ''' 
        n = 2 ** (num_bits - 1) - 1 
        
        scale = max(w_min.abs(), w_max.abs()) 
        scale = torch.clamp(scale, min = 1e-8) / n 
        
    return scale 

def symmetric_linear_quantization_params(num_bits,
                                         saturation_min,
                                         saturation_max,
                                         per_channel=False):
    """
    Compute the scaling factor and zeropoint with the given quantization range for symmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    per_channel: if True, calculate the scaling factor per channel.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n
        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n

    return scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range for asymmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    integral_zero_point: if True, adjust zero_point accordingly to make sure 0.0 in floating point tensor
                         be exactly mapped to an integer value.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** num_bits - 1
        scale = torch.clamp((saturation_max - saturation_min), min=1e-8) / float(n)

        # For asymmetric quantization, the current hardware support scaled unsigned integers without zero_point.
        # So saturation_min = 0 (we only do asymmetric quantization for activations after ReLU.)
        zero_point = -saturation_min / scale

        if integral_zero_point:
            if isinstance(zero_point, torch.Tensor):
                zero_point = zero_point.round()
            else:
                zero_point = float(round(zero_point))

        return scale, zero_point


def batch_frexp(inputs):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """
    shape_of_input = inputs.size()

    # transform the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())

    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2 ** 31)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = 31. - output_e

    return torch.from_numpy(output_m).cuda().view(shape_of_input), \
           torch.from_numpy(output_e).cuda().view(shape_of_input)


class ste_round(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        return grad_output.clone()
        ''' 
        return grad_output 
    
class sparse_pass(Function): 
    """ 
    convert gradient from sparse to dense 
    """ 
    
    @staticmethod 
    def forward(ctx, x): 
        return x 
    
    @staticmethod 
    def backward(ctx, grad_output): 
        print(grad_output.type()) 
        return grad_output.to_dense() 


class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None, backwardpass = False): 
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k - 1) - 1

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0.).cuda() 
        if backwardpass: # add the conditional checking 
            new_quant_x = linear_quantize(x, scale, zero_point, inplace = True) 
            ctx.scale = scale 
            return new_quant_x 
        else: 
            new_quant_x = linear_quantize(x, scale, zero_point, inplace = False) 
        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else: 
            scale = scale.view(-1) 

        '''
        return grad_output.clone() / scale, None, None, None 
        ''' 
        return grad_output / scale, None, None, None, None 


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using asymmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None, specified_zero_point=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of AsymmetricQuantFunction requires pre-calculated scaling factor.
        The current hardware support requires asymmetric quantization to use scaled unsigned integers
        without zero_point, so asymmetric quantization is for activations after ReLU, and zero_point is set to 0.
        specified_scale: pre-calculated scaling factor for the tensor x
        specified_zero_point: pre-calculated zero_point for the tensor x
        """
        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The AsymmetricQuantFunction requires a pre-calculated scaling factor")

        if specified_zero_point is not None:
            zero_point = specified_zero_point
        else:
            zero_point = torch.tensor(0).cuda()

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2 ** k - 1

        new_quant_x = torch.clamp(new_quant_x, 0, n)

        ctx.scale = scale

        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)
        return grad_output.clone() / scale, None, None, None


class transfer_float_averaging_to_int_averaging(Function):
    """
    Straight-through Estimator(STE) for Int Averaging

    The eps is used to avoid pytorh representation error like 2 = 1.99999999
    However, the eps has upper bound,
    take 7x7 integer average pooling as an example, the eps should be chosen to satisfy 48/49 + eps < 1.
    """

    @staticmethod
    def forward(ctx, x, eps=0.01):
        with torch.no_grad():
            x_int = torch.trunc(x + eps)
            return x_int

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class fixedpoint_fn(Function):
    """
    Function to perform fixed-point arthmetic that can match integer arthmetic on hardware.

    Parameters:
    ----------
    z: input tensor
    bitwidth: quantization bitwidth
    quant_mode: The mode for quantization, 'symmetric' or 'asymmetric'.
    z_scaling_factor: scaling factor of tensor z
    case: case 0: z = Wx, case 1: z = (W_y) y + (W_x) x
    pre_act_scaling_factor: the scaling factor of the previous activation quantization layer
    pre_weight_scaling_factor: the scaling factor of the previous weight quantization layer
    identity: the output tensor of the identity branch
    identity_scaling_factor: the scaling factor of the previous activation quantization of identity
    identity_weight_scaling_factor: the scaling factor of the weight quantization layer in the identity branch
    """

    @staticmethod
    def forward(ctx, z, bitwidth, quant_mode, z_scaling_factor, case, pre_act_scaling_factor=None,
                pre_weight_scaling_factor=None, identity=None, identity_scaling_factor=None,
                identity_weight_scaling_factor=None):
        if quant_mode == 'symmetric':
            n = 2 ** (bitwidth - 1) - 1
        else:
            n = 2 ** bitwidth - 1

        with torch.no_grad():
            # reshape all the tensors to have correct sizes.
            if len(z.shape) == 4:
                z_scaling_factor = transfer_conv_size(z_scaling_factor)
                pre_act_scaling_factor = transfer_conv_size(pre_act_scaling_factor)
                pre_weight_scaling_factor = transfer_conv_size(pre_weight_scaling_factor)
            elif len(z.shape) == 2:
                z_scaling_factor = transfer_fc_size(z_scaling_factor)
                pre_act_scaling_factor = transfer_fc_size(pre_act_scaling_factor)
                pre_weight_scaling_factor = transfer_fc_size(pre_weight_scaling_factor)
            ctx.z_scaling_factor = z_scaling_factor
            if case == 1:
                if len(z.shape) == 4:
                    identity_scaling_factor = transfer_conv_size(identity_scaling_factor)
                    identity_weight_scaling_factor = transfer_conv_size(identity_weight_scaling_factor)
                elif len(z.shape) == 2:
                    identity_scaling_factor = transfer_fc_size(identity_scaling_factor)
                    identity_weight_scaling_factor = transfer_fc_size(identity_weight_scaling_factor)

            if case == 0:
                # convert z from floating point to integer first
                z_int = torch.round(z / pre_act_scaling_factor / pre_weight_scaling_factor)
                # follow TVM's computation
                _A = (pre_act_scaling_factor.type(torch.double) * pre_weight_scaling_factor.type(torch.double))
                _B = (_A.type(torch.float)).type(torch.double)
                _C = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _B / _C

                if len(z.shape) == 4:
                    new_scale = transfer_conv_size(new_scale)
                elif len(z.shape) == 2:
                    new_scale = transfer_fc_size(new_scale)

                m, e = batch_frexp(new_scale)

                output = z_int.type(torch.double) * m.type(torch.double)

                output = torch.round(output / (2.0 ** e))

                if quant_mode == 'symmetric':
                    return torch.clamp(output.type(torch.float), -n - 1, n)
                else:
                    return torch.clamp(output.type(torch.float), 0, n)

            # in case 1, the tensor z should be separeted into 2 parts, one from wy, another from wx
            elif case == 1:
                wx_int = torch.round(identity / identity_scaling_factor / identity_weight_scaling_factor)

                _A = (identity_scaling_factor.type(torch.double) * identity_weight_scaling_factor.type(torch.double))
                _B = (_A.type(torch.float)).type(torch.double)
                _C = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _B / _C

                if len(z.shape) == 4:
                    new_scale = transfer_conv_size(new_scale)
                elif len(z.shape) == 2:
                    new_scale = transfer_fc_size(new_scale)

                m1, e1 = batch_frexp(new_scale)

                output1 = wx_int.type(torch.double) * m1.type(torch.double)

                output1 /= (2.0 ** e1)
                output1 = torch.round(output1)

                wy = (z - identity)
                wy_int = torch.round(wy / pre_act_scaling_factor / pre_weight_scaling_factor)

                _A = (pre_act_scaling_factor.type(torch.double) * pre_weight_scaling_factor.type(torch.double))
                _B = (_A.type(torch.float)).type(torch.double)
                _C = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _B / _C

                if len(z.shape) == 4:
                    new_scale = transfer_conv_size(new_scale)
                elif len(z.shape) == 2:
                    new_scale = transfer_fc_size(new_scale)

                m2, e2 = batch_frexp(new_scale)

                output2 = wy_int.type(torch.double) * m2.type(torch.double)

                output2 /= (2.0 ** e2)
                output2 = torch.round(output2)

                return (output1 + output2).type(torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, None, None, None, None, None
