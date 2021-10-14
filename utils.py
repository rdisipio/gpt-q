import torch
import numpy as np

from torch.autograd import Variable


def make_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_padding_mask(src, pad=0):
    mask = (src != pad).unsqueeze(-2)
    return mask


def make_lookahead_mask(tgt, pad=0):
    "Create a mask to hide padding and future words."
    tgt_mask = make_padding_mask(tgt, pad)
    tgt_mask = tgt_mask & Variable(
        make_subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
