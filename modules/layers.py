import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from modules.helpers import sequence_mask, masked_normalization_inf, \
    masked_normalization
from modules.initializations import _init_tensor


class GaussianNoise(nn.Module):
    def __init__(self, stddev, mean=.0):
        """
        Additive Gaussian Noise layer
        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean

    def forward(self, x, s):
        if self.training:
            dist = Normal(self.mean, self.stddev)
            return x + dist.sample()
        return x

    def __repr__(self):
        return '{} (mean={}, stddev={})'.format(self.__class__.__name__,
                                                str(self.mean),
                                                str(self.stddev))


class Bridge(nn.Module):
    def __init__(self,
                 encoder_size,
                 decoder_size,
                 bidirectional,
                 rnn_type,
                 non_linearity=None):
        super().__init__()

        # todo: learn different W for each layer to layer mapping

        self.non_linearity = non_linearity
        self.bidirectional = bidirectional

        number_of_states = 2 if rnn_type == "LSTM" else 1
        self.bridge = nn.ModuleList([nn.Linear(encoder_size, decoder_size)
                                     for _ in range(number_of_states)])

        self.init_weights()

    def init_weights(self):
        pass

    @staticmethod
    def _fix_hidden(_hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        fwd_final = _hidden[0:_hidden.size(0):2]
        bwd_final = _hidden[1:_hidden.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)
        return final

    def _bottle_hidden(self, linear, states):
        result = linear(states)

        if self.non_linearity == "tanh":
            result = torch.tanh(result)
        elif self.non_linearity == "relu":
            result = F.relu(result)

        return result

    def forward(self, hidden):
        """Forward hidden state through bridge."""

        if isinstance(hidden, tuple):  # LSTM
            # concat directions
            if self.bidirectional:
                hidden = tuple(self._fix_hidden(h) for h in hidden)
            outs = tuple([self._bottle_hidden(state, hidden[ix])
                          for ix, state in enumerate(self.bridge)])
        else:
            outs = self._bottle_hidden(self.bridge[0], hidden)

        return outs


class RNNBridge(nn.Module):

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 dec_layers: int,
                 dec_type: str,
                 interaction: bool = False,
                 activation: str = "tanh",
                 **kwargs):
        super(RNNBridge, self).__init__()

        self.dec_type = dec_type
        self.interaction = interaction
        self.activation = activation

        if self.interaction:
            self.fc_inter = nn.Linear(enc_dim, enc_dim)

        # hidden state's bridge
        self.B_h = nn.ModuleList([nn.Linear(enc_dim, dec_dim)
                                  for _ in range(dec_layers)])

        # cell state's bridge
        if self.dec_type == "LSTM":
            self.B_c = nn.ModuleList([nn.Linear(enc_dim, dec_dim)
                                      for _ in range(dec_layers)])

    def forward(self, x):

        if self.interaction:
            x = self.fc_inter(x)
            if self.activation == "relu":
                x = torch.relu(x)
            elif self.activation == "tanh":
                x = torch.tanh(x)
            else:
                raise NotImplementedError

        h0 = torch.cat([b(x).unsqueeze(0) for b in self.B_h])

        if self.dec_type == "LSTM":
            c0 = torch.cat([b(x).unsqueeze(0) for b in self.B_c])
            return h0, c0
        else:
            return h0


class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 noise=.0,
                 emb_dropout=.0,
                 trainable=True,
                 grad_mask=None,
                 layer_norm=False,
                 max_norm=None,
                 padding_idx=0,
                 scale=False):
        super(Embed, self).__init__()

        self.layer_norm = layer_norm
        self.trainable = trainable
        self.max_norm = max_norm
        self.noise = noise
        self.scale = scale

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      # max_norm=max_norm,
                                      padding_idx=padding_idx)
        self.embedding.weight.requires_grad = self.trainable

        # the dropout "layer" for the word embeddings
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.grad_mask = grad_mask

        if self.layer_norm:
            self.LN_e = nn.LayerNorm(embedding_dim, eps=1e-6)

        if self.grad_mask is not None:
            self.set_grad_mask(self.grad_mask)

        self.embedding_dim = self.embedding.embedding_dim

        # self.init_weights()

    def init_weights(self):
        self.embedding.weight.requires_grad = self.trainable
        _init_tensor(self.embedding.weight, "normal",
                     self.embedding.embedding_dim)
        torch.nn.init.constant_(self.embedding.weight[0], 0)

    def transfer_weights(self, embeddings):
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float(), not self.trainable,
            max_norm=self.max_norm, padding_idx=0)
        self.embedding.weight.data[0].zero_()

    def _emb_hook(self, grad):
        return grad * self.grad_mask.unsqueeze(1).type_as(grad)

    def set_grad_mask(self, mask):
        self.grad_mask = torch.from_numpy(mask)
        self.embedding.weight.register_hook(self._emb_hook)

    def regularize(self, embeddings):
        embeddings = self.emb_dropout(embeddings)
        return embeddings

    def expectation(self, dists=None, logits=None):
        """
        Obtain a weighted sum (expectation) of all the embeddings, from a
        given probability distribution.

        """
        # raise NotImplementedError  # rethink about adding regularization here

        if dists is None and logits is not None:
            batch, length, dim = logits.size()
            flat_probs = F.softmax(logits, -1).view(batch * length, dim)
        elif dists is not None and logits is None:
            batch, length, dim = dists.size()
            flat_probs = dists.view(batch * length, dim)
        else:
            raise ValueError

        flat_embs = flat_probs.mm(self.embedding.weight)
        embs = flat_embs.view(batch, length, flat_embs.size(1))

        if self.scale:
            embs = embs * math.sqrt(self.embedding_dim)

        # apply layer normalization on the expectation
        if self.layer_norm:
            embs = self.LN_e(embs)

        # apply all embedding layer's regularizations
        embs = self.regularize(embs)

        return embs

    def forward(self, x):
        embeddings = self.embedding(x)

        if self.scale:
            embeddings = embeddings * math.sqrt(self.embedding_dim)

        if self.layer_norm:
            embeddings = self.LN_e(embeddings)

        embeddings = self.regularize(embeddings)

        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=True,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequence, lengths):

        energies = self.attention(sequence).squeeze()

        # construct a mask, based on sentence lengths
        mask = sequence_mask(lengths, energies.size(1))

        # scores = masked_normalization_inf(energies, mask)
        scores = masked_normalization(energies, mask)

        contexts = (sequence * scores.unsqueeze(-1)).sum(1)

        return contexts, scores


class Attention(nn.Module):
    def __init__(self,
                 input_size,
                 context_size,
                 batch_first=True,
                 layer_norm=False,
                 non_linearity="tanh",
                 method="general",
                 coverage=False):
        super(Attention, self).__init__()

        self.batch_first = batch_first
        self.method = method
        self.coverage = coverage
        self.layer_norm = layer_norm

        if self.method not in ["dot", "general", "concat", "additive"]:
            raise ValueError("Please select a valid attention type.")

        if self.coverage:
            self.W_c = nn.Linear(1, context_size, bias=False)
            self.method = "additive"

        if non_linearity == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        if self.method == "general":
            self.W_h = nn.Linear(input_size, context_size, bias=False)

        elif self.method == "additive":
            self.W_h = nn.Linear(input_size, context_size, bias=False)
            self.W_s = nn.Linear(context_size, context_size, bias=False)
            self.W_v = nn.Linear(context_size, 1, bias=False)

            if self.layer_norm:
                self.W_v_norm = nn.LayerNorm(context_size, eps=1e-6)

        elif self.method == "concat":
            self.W_h = nn.Linear(input_size + context_size,
                                 context_size, bias=False)
            self.W_v = nn.Linear(context_size, 1, bias=False)

            if self.layer_norm:
                self.W_v_norm = nn.LayerNorm(context_size, eps=1e-6)

        self.enc_proj = None

        self.init_weights()

    def init_weights(self):
        pass

    def score(self, sequence, query, coverage=None):
        batch_size, max_length, feat_size = sequence.size()

        if self.method == "dot":
            energies = torch.bmm(sequence, query.unsqueeze(2)).squeeze(2)

        elif self.method == "additive":
            assert self.enc_proj is not None
            dec = self.W_s(query)
            sums = self.enc_proj + dec.unsqueeze(1)

            if self.coverage:
                cov = self.W_c(coverage.unsqueeze(-1))
                sums = sums + cov

            energies = self.W_v(self.activation(sums)).squeeze(2)

        elif self.method == "general":
            assert self.enc_proj is not None
            energies = torch.bmm(self.enc_proj, query.unsqueeze(2)).squeeze(2)

        elif self.method == "concat":
            c = query.unsqueeze(1).expand(-1, max_length, -1)
            u = self.W_h(torch.cat([sequence, c], -1))
            energies = self.W_v(self.activation(u)).squeeze(2)

        else:
            raise ValueError

        return energies

    def precompute_enc_projections(self, enc):
        self.enc_proj = self.W_h(enc)

    def forward(self, sequence, query, mask, coverage=None):

        batch, seq_length, _ = sequence.size()

        energies = self.score(sequence, query, coverage)

        scores = masked_normalization_inf(energies, mask)
        # scores = self.masked_normalization(energies, mask)

        # contexts = torch.bmm(scores.unsqueeze(1), sequence).squeeze(1)
        contexts = (sequence * scores.unsqueeze(-1)).sum(1).unsqueeze(1)
        # contexts = (scores.unsqueeze(1) @ sequence)

        return contexts, scores


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
