import torch
import numpy as np


def make_src_mask(sz):
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask


def make_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_padding_mask(src, pad=0):
    src = torch.from_numpy(src)
    mask = (src != pad)
    mask = torch.unsqueeze(mask, -2)
    return mask


def make_lookahead_mask(tgt, pad=0):
    "Create a mask to hide padding and future words."
    tgt_mask = make_padding_mask(tgt, pad)
    tgt_mask = tgt_mask & Variable(
        make_subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
