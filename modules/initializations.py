import math

import torch
from scipy.stats import truncnorm
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def plot_norm(w):
    import matplotlib.pyplot as plt
    x = torch.norm(w, dim=1)
    plt.hist(x.detach().numpy(), normed=True, bins=30)
    plt.show()


def normal_trunc_(tensor, stdv):
    _samples = torch.from_numpy(truncnorm.rvs(-3, 3, size=tensor.size()))
    _samples *= stdv
    tensor.data.copy_(_samples)


def _init_tensor(tensor, init, fan_in=None):
    if fan_in is None:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    else:
        fan_in = fan_in

    # if fan_in == fan_out:
    #     init = "orthogonal"

    if init == "normal":
        # sample from a truncated normal, to avoid sampling outliers
        stdv = 1.0 / math.sqrt(fan_in)
        normal_trunc_(tensor.contiguous(), stdv)
    if init == "transformer":
        # sample from a truncated normal, to avoid sampling outliers
        stdv = tensor.size(1) ** -0.5
        normal_trunc_(tensor.contiguous(), stdv)
    elif init == "uniform":
        stdv = math.sqrt(3.0 / fan_in)
        torch.nn.init.uniform_(tensor.contiguous(), 0, stdv)
    elif init == "xavier_uniform":
        torch.nn.init.xavier_uniform_(tensor.contiguous())
    elif init == "xavier_normal":
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        stdv = math.sqrt(2.0 / float(fan_in + fan_out))
        normal_trunc_(tensor.contiguous(), stdv)
    elif init == "orthogonal":
        torch.nn.init.orthogonal_(tensor.contiguous())


def _init_lstm(module, lstm_hh_init, lstm_ih_init):
    for _name, _param in module.named_parameters():
        n = _param.size(0) // 4

        if "bias" in _name:
            # ingate, forgetgate, cellgate, outgate
            # gates are (b_hi|b_hf|b_hg|b_ho) of shape (4*hidden_size)
            _param.data.fill_(0.)
            _param.data[n:n * 2].fill_(1.)

        elif "_ih_" in _name:
            _init_tensor(_param.data[0:n], init=lstm_ih_init)
            _init_tensor(_param.data[n:n * 2], init=lstm_ih_init)
            _init_tensor(_param.data[n * 2:n * 3], init=lstm_ih_init)
            _init_tensor(_param.data[n * 3:n * 4], init=lstm_ih_init)

        elif "_hh_" in _name:
            _init_tensor(_param.data[0:n], init=lstm_hh_init)
            _init_tensor(_param.data[n:n * 2], init=lstm_hh_init)
            _init_tensor(_param.data[n * 2:n * 3], init=lstm_hh_init)
            _init_tensor(_param.data[n * 3:n * 4], init=lstm_hh_init)


def model_init(model,
               linear_init="xavier_uniform",
               lstm_hh_init="orthogonal",
               lstm_ih_init="xavier_uniform",
               embed_init="xavier_uniform"):
    with torch.no_grad():

        for m_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                _init_tensor(module.weight, linear_init)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

            if isinstance(module, nn.LSTM):
                _init_lstm(module, lstm_hh_init, lstm_ih_init)

        for m_name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                _init_tensor(module.weight, embed_init, module.embedding_dim)
                if module.padding_idx is not None:
                    torch.nn.init.zeros_(module.weight[module.padding_idx])
