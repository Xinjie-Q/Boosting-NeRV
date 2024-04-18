from pyexpat.errors import messages
import torch
import math
from torch.autograd import Function
import constriction
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder

def get_np_size(x):
    return x.size * x.itemsize

class DiffEntropyModel():
    def __init__(self, distribution="gaussian"):
        self.distribution = distribution

    def cal_bitrate(self, code, quant, training):
        return self.cal_global_bitrate(code, quant, training)

    def cal_global_bitrate(self, code, quant, training):
        mean = torch.mean(code)
        std = torch.std(code)
        if training:
            noise = torch.empty_like(code).uniform_(-0.5, 0.5)
            code = code+noise
            real_bits = 0
        else:
            code = quant
            real_bits = compress_matrix_flatten_gaussian_global(code, mean, std)
        bits = torch.sum(self.get_bits(code, mean, std))
        return {"bitrate": bits, "mean":mean, "std":std, "real_bitrate":real_bits}

    def get_bits(self, x, mu, sigma):
        sigma = sigma.clamp(1e-5, 1e10)
        if self.distribution == "gaussian":
            gaussian = torch.distributions.normal.Normal(mu, sigma)
        else:
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits


def compress_matrix_flatten_gaussian_global(matrix, mean, std):
    '''
    :param matrix: tensor
    :return compressed, symtable
    '''
    mean = mean.item()
    std = std.clamp(1e-5, 1e10).item()
    min_value, max_value = matrix.min().int().item(), matrix.max().int().item()
    if min_value == max_value:
        max_value = min_value + 1
    message = np.array(matrix.int().flatten().tolist(), dtype=np.int32)
    entropy_model = constriction.stream.model.QuantizedGaussian(min_value, max_value, mean, std)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    total_bits = get_np_size(compressed) * 8
    return total_bits


def compress_matrix_flatten_categorical(matrix):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix) #matrix.flatten()
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judege_type(min_value, max_value))
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    return compressed, unique_counts, unique

def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


