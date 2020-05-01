import torch
from torch import nn
from torch.nn import functional as F

from modules.helpers import sequence_mask, avg_vectors, id2dist


class SequenceLoss(nn.Module):
    """
    Compute a sequence loss (i.e. per timestep).
    Used for tasks such as Translation, Language Modeling and
    Sequence Labelling.
    """

    def __init__(self, tgt_vocab_size, device, ignore_index=0,
                 label_smoothing=0):
        super(SequenceLoss, self).__init__()

        assert 0.0 <= label_smoothing <= 1.0
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        if self.label_smoothing > 0:

            if self.ignore_index >= 0:
                low_confidence = label_smoothing / (tgt_vocab_size - 2)
            else:  # there is no padding id
                low_confidence = label_smoothing / (tgt_vocab_size - 1)

            self.one_hot = torch.full((tgt_vocab_size,), low_confidence)

            if self.ignore_index >= 0:
                self.one_hot[self.ignore_index] = 0

            self.one_hot = self.one_hot.to(device)
            self.high_confidence = 1.0 - label_smoothing
        else:
            self.high_confidence = None

    def kl_loss(self, logits, labels, lengths):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """

        _logits = logits.contiguous().view(-1, logits.size(-1))
        _labels = labels.contiguous().view(-1)

        log_prob = F.log_softmax(_logits, dim=1)

        model_prob = self.one_hot.repeat(_labels.size(0), 1)
        model_prob.scatter_(1, _labels.unsqueeze(1), self.high_confidence)

        losses = F.kl_div(log_prob, model_prob, reduction='none')

        mask = sequence_mask(lengths, labels.size(1)).view(-1).float()
        losses = losses.sum(1) * mask
        loss = losses.sum() / mask.sum()

        return loss, losses

    def cross_entropy_loss(self, logits, labels, lengths=None):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        _logits = logits.contiguous().view(-1, logits.size(-1))

        if self.ignore_index >= 0:
            _labels = labels.contiguous().view(-1)
        else:
            assert lengths is not None
            mask = ~sequence_mask(lengths, labels.size(1))
            _labels = labels.masked_fill_(mask, -1).contiguous().view(-1)

        if lengths is None:
            loss = F.cross_entropy(_logits, _labels,
                                   ignore_index=self.ignore_index)
            return loss

        else:
            _loss = F.cross_entropy(_logits, _labels,
                                    ignore_index=self.ignore_index,
                                    reduction='none')
            _loss_per_step = _loss.view(labels.size())
            loss = _loss.sum() / lengths.float().sum()
            return loss, _loss_per_step

    def forward(self, logits, labels, lengths=None):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """

        if self.label_smoothing > 0:
            return self.kl_loss(logits, labels, lengths)
        else:
            return self.cross_entropy_loss(logits, labels, lengths)


def compute_kernel(x, y):
    """Computes a positive definite kernel function between x and y.
    Args:
        x(torch.FloatTensor): [-1, x_dim] tensor.
        y(torch.FloatTensor): [-1, y_dim] tensor.
    Returns:
        torch.FloatTensor: kernel between x and y of [x_dim, y_dim].
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    """(Biased) Estimator of the Maximum Mean Discrepancy two sets of samples."""
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def length_loss(logits, lengths, eos):
    """
    Length control loss, using a sequence of length labels (with eos token).

    Args:
        logits:
        lengths:
        eos:

    Returns:

    """
    mask = sequence_mask(lengths - 1, lengths.max())
    eos_labels = ((1 - mask) * eos).long().contiguous().view(-1)

    _logits = logits.contiguous().view(-1, logits.size(-1))
    loss = F.cross_entropy(_logits, eos_labels, ignore_index=0)

    return loss


def pairwise_loss(a, b, dist="cosine"):
    if dist == "euclidean":
        return F.pairwise_distance(a, b).mean()
    elif dist == "cosine":
        return 1 - F.cosine_similarity(a, b).mean()
    elif dist == "dot":
        dot = torch.bmm(a.unsqueeze(1), b.unsqueeze(-1)).squeeze()
        scaled_dot = dot.mean() / a.size(1)
        raise NotImplementedError
    else:
        raise ValueError


def centroid_loss(enc_feats, dec_feats, src_lengths, trg_lengths,
                  enc_scores=None,
                  dec_scores=None,
                  distance="cosine",
                  pool_func="mean",
                  mapping: torch.Tensor = None,
                  **kwargs):
    """
    Compute the pairwise distance of various outputs of the seq^3 architecture.
    """

    enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
    dec_mask = sequence_mask(trg_lengths).unsqueeze(-1).float()

    # Aggregate the vectors of each sequence
    if pool_func == "mean":
        x_emb, _ = avg_vectors(enc_feats, enc_mask, enc_scores)
        y_emb, _ = avg_vectors(dec_feats, dec_mask, dec_scores)
    elif pool_func == "max":
        x_emb = enc_feats.max(1)[0]
        y_emb = dec_feats.max(1)[0]
    elif pool_func == "sum":
        x_emb = enc_feats.sum(1)
        y_emb = dec_feats.sum(1)
    else:
        raise ValueError

    # Apply a rotation operation on the source embedding
    if mapping is not None:
        x_emb = torch.matmul(x_emb, mapping)

    return pairwise_loss(x_emb, y_emb, distance)


def masked_kld(inp_logits, trg_logits, lengths, tau=1, mask_ids=[]):
    """
    Compute the grounding loss using a pretrained "oracle" LM.
    The loss is computed using the produced posteriors over the vocabulary
    produced by a generator and the posteriors of the "oracle" LM.

    Args:
        logits: the logits of the generator
        words: the argmax of the logits
        oracle: the oracle LM
        tau: the temperature of the softmax
        lengths: the lengths of the target sequence. Used for masking the loss.


    Debug = -F.softmax(_logits, -1) * torch.log(F.softmax(logits, -1) /
                                                F.softmax(_logits, -1))

    Returns:
        the average KL Divergence per timestep (word)

    """

    input_logp = F.log_softmax(inp_logits / tau, -1)
    target_p = F.softmax(trg_logits / tau, -1)

    # zero padded timesteps
    mask = sequence_mask(lengths).unsqueeze(-1).float()

    # shape: batch x seq_length x tokens
    loss = F.kl_div(input_logp * mask, target_p * mask, reduction='none')

    for i in mask_ids:
        loss[:, :, i] = 0

    # sum over words/vocab (KL per word/timestep !)
    loss = loss.sum(-1)

    loss = loss * mask.squeeze()
    total_loss = loss.sum() / mask.sum()

    return total_loss, loss


def masked_mse(inp_logits, trg_logits, lengths, mask_ids=[]):
    # zero padded timesteps
    mask = sequence_mask(lengths).unsqueeze(-1).float()

    # shape: batch x seq_length x tokens
    loss = F.mse_loss(inp_logits * mask, trg_logits * mask, reduction='none')

    for i in mask_ids:
        loss[:, :, i] = 0

    loss = loss.mean(-1)
    loss = loss * mask.squeeze()
    total_loss = loss.sum() / mask.sum()

    return total_loss, loss


def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)


def dot3D(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return bdot(a.view(B * S, -1), b.view(B * S, -1)).view(B, S)


def discrete_samples(sos_ids, sample_ids):
    # prepend the <sos> id and remove the last output
    prior_inp_ids = torch.cat((sos_ids, sample_ids[:, :-1]), 1)
    return prior_inp_ids


def differentiable_samples(embedding, dists, sos_id=None):
    if sos_id is not None:
        sos_ids = torch.empty(dists.size(0), 1, dtype=torch.long,
                              device=dists.device).fill_(sos_id)

        # prepend the <sos> symbol and remove the last output
        sos_dist = id2dist(sos_ids, dists.size(-1)).unsqueeze(1)
        sos_dist = torch.cat((sos_dist, dists[:, :-1, :]), 1)

    # get the expected embeddings over the dists
    e_i = embedding.expectation(sos_dist)
    return e_i


def prior_loss(outputs, trg_len, prior, mode, sos_id=1, tau=1, init_h=None):
    # The actual tokens that were used during generating the target seq.
    # When the decoder is trained with 100% teacher forcing,
    # sampled_tokens == trg_inp
    # sample_ids = outputs["dists"].max(-1)[1]
    prior_inps = differentiable_samples(prior.encoder.embed,
                                        outputs["dists"], sos_id)

    if mode == "prior":
        lm_outs = prior(prior_inps, trg_len, init_h)
        loss, loss_i = masked_kld(outputs["logits"], lm_outs["logits"],
                                  trg_len,
                                  tau=tau, mask_ids=[0, 1, 2, 3])

    elif mode == "discriminator":
        # feed the embeddings to the LM Discriminator
        lm_outs = prior(prior_inps, trg_len, init_h)
        mask = sequence_mask(trg_len).float()

        # check = F.cross_entropy(
        #     lm_outs["logits"].contiguous().view(-1, lm_outs["logits"].size(-1)),
        #     outputs["dists"].argmax(-1).view(-1), ignore_index=0,
        #     reduction='none')

        prior_log_probs = F.log_softmax(lm_outs["logits"], -1)
        loss_i = dot3D(outputs["dists"].contiguous(),
                       prior_log_probs.contiguous()) * mask

        cross_entropy = loss_i.sum() / mask.sum()

        # avoid collapse
        # agg_logits = outputs["logits"].sum(1) / mask.sum(-1, keepdim=True)
        # entropy = Categorical(logits=agg_logits).entropy().mean()

        loss = - cross_entropy

    else:
        raise ValueError

    return loss, loss_i, lm_outs["logits"]
