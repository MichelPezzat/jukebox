import numpy as np
import sys
import math
import time

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.autograd import Variable


def build_log_p_T(args, train, val):
    T_hist = torch.zeros(100000)
    max_T = 0
    for ex in train.examples+val.examples:
        ex_len = len(ex.text)
        T_hist[ex_len] += 1
        if ex_len > max_T:
            max_T = ex_len

    if args.indep_bernoulli:
        max_T = int(max_T*1.25)
        T_hist += 1

    T_hist = T_hist[:max_T+1]
    log_p_T = torch.log(T_hist/T_hist.sum())

    return log_p_T, max_T

def get_kl_weight(hps, i):
    if hps.initial_kl_zero == 0 and hps.kl_rampup_time == 0:
        return 1.0, True

    x_start = hps.initial_kl_zero
    x_end = hps.initial_kl_zero + hps.kl_rampup_time
    y_start = 0.00001
    y_end = 1.0
    done = False
    if i < x_start:
        cur_kl_weight = y_start
    elif i > x_end:
        cur_kl_weight = y_end
        done = True
    else:
        cur_kl_weight = (i-x_start)/(x_end-x_start)*(y_end-y_start) + y_start

    return cur_kl_weight, done

# Model utility functions
# ------------------------------------------------------------------------------------------------------------------------------

def make_pos_cond(T, B, lengths, max_T):
    device = lengths.device

    p_plus_int = torch.arange(T, device=device)[:, None].repeat(1, B)[:, :, None]
    p_plus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_plus_oh.scatter_(2, p_plus_int, 1)
    
    p_minus_int = lengths[None, :] - 1 - torch.arange(T, device=device)[:, None]
    p_minus_int[p_minus_int < 0] = max_T-1
    p_minus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_minus_oh.scatter_(2, p_minus_int[:, :, None], 1)
    
    pos_cond = torch.cat((p_plus_oh, p_minus_oh), -1) # [T, B, max_T*2]

    return pos_cond

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    if batch_first:
        inputs = inputs.transpose(0, 1)

    if inputs.size(1) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')

    reversed_inputs = inputs.data.clone()
    for i, length in enumerate(lengths):
        time_ind = torch.LongTensor(list(reversed(range(length))))
        reversed_inputs[:length, i] = inputs[:, i][time_ind]

    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
        
    return reversed_input
