import torch
from numpy import mean
from torch.nn import functional as F


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1)))


def future_mask(batch):
    """
    Used for Transformer-based self-attention, for masking the
    future tokens
    :param batch:
    :return:
    """
    sz = batch.size(0)
    mask = torch.triu(torch.ones(sz, sz, device=batch.device)) == 1
    mask = mask.transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def masked_normalization(logits, mask):
    scores = F.softmax(logits, dim=-1)

    # apply the mask - zero out masked timesteps
    masked_scores = scores * mask.float()

    # re-normalize the masked scores
    normed_scores = masked_scores.div(masked_scores.sum(-1, keepdim=True))

    return normed_scores


def masked_mean(vecs, mask):
    masked_vecs = vecs * mask.float()

    mean = masked_vecs.sum(1) / mask.sum(1)

    return mean


def masked_normalization_inf(logits, mask):
    logits.masked_fill_(~mask, float('-inf'))
    # energies.masked_fill_(~mask, -1e18)

    scores = F.softmax(logits, dim=-1)

    return scores


def expected_vecs(dists, vecs):
    flat_probs = dists.view(dists.size(0) * dists.size(1),
                            dists.size(2))
    flat_embs = flat_probs.mm(vecs)
    embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))
    return embs


def relax_softmax(logits, tau=1, gumbel=False, hard=False, beta=1, dim=-1,
                  eps=1e-10):
    if gumbel:

        # ~Gumbel(0,beta)
        # u = torch.empty_like(logits).uniform_()
        # gumbels = - beta * torch.log(eps - torch.log(u + eps))

        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = gumbels * beta

        if not isinstance(tau, (int, float)):
            tau = tau.unsqueeze(-1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

    else:
        y_soft = F.softmax(logits / tau, dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def infer_sample_lengths(samples, eos_id=2):
    eos_ids = samples == eos_id
    eos_ids[:, -1] = True
    nonz = (eos_ids.int() > 0)
    _lengths = ((nonz.cumsum(1) == 1) & nonz).int().max(1)[1] + 1
    return _lengths


def first_nonzero_index(x, axis=0):
    nonz = (x.int() > 0).int()
    return ((nonz.cumsum(axis) == 1) & nonz).int().argmax(axis)


def infer_seq_lengths(latent_dists, x_lengths, eos_id=2):
    # 2 is the EOS id
    # todo: the EOS id is hardcoded. Turn it into a parameter
    eos_ids = latent_dists.argmax(-1) == eos_id
    _lengths = first_nonzero_index(eos_ids, 1) + 1

    # the predicted length has to be:
    # - at most 1.5 the input length
    # - at least half the input length
    _upper = (x_lengths.float() * 1.3).long()
    _lower = (x_lengths.float() * 0.7).long()

    # if the model predicts lengths larger than the upper bound clip them
    _lengths = torch.min(_lengths, _upper)

    # if the model predicts lengths shorter than the lower bound clip them
    _lengths = torch.max(_lengths, _lower)

    return _lengths


def model_tied(model):
    params = {}
    for na, pa in model.state_dict().items():
        shared = []
        for nb, pb in model.state_dict().items():
            try:
                if bool(torch.all(torch.eq(pa, pb))) and na != nb:
                    shared.append(nb)
            except:
                pass
        if len(shared) > 0:
            params[na] = shared
    return params


def params2tree(ls, trim=True):
    tree = {}

    for item in ls:
        t = tree
        for part in item.split('.'):
            t = t.setdefault(part, {})

    def foo(d):

        return {
            key: foo(value)
            for key, value in d.items() if len(value) > 0
        }

    if trim:
        tree = foo(tree)

    return tree


def model_weights(module, tree={}, parent=""):
    for n, m in module.named_children():

        if len(parent) > 0:
            name = ".".join([parent, n])
        else:
            name = n

        if hasattr(m, 'weight'):
            tree[name] = m.weight
            return tree

        else:
            model_weights(m, tree, name)


def id2dist(ids, k):
    _hard = torch.zeros((ids.size(0), k),
                        layout=ids.layout, device=ids.device)
    y_hard = _hard.scatter_(-1, ids.view(-1, 1), 1.0)
    return y_hard


def coin_flip(prob):
    """
    Return the outcome of a biased coin flip.
    Args:
        prob: the probability of True.

    Returns: bool

    """
    return prob > 0 and torch.rand(1).item() < prob


def avg_vectors(vectors, mask, energies=None):
    if energies is None:
        centroid = masked_mean(vectors, mask)
        return centroid, None
    else:
        masked_scores = energies * mask.float()
        normed_scores = masked_scores.div(masked_scores.sum(1, keepdim=True))
        centroid = (vectors * normed_scores).sum(1)
    return centroid, normed_scores


def var_vectors(vectors, mask, energies=None):
    masked_scores = energies * mask.float()
    normed_scores = masked_scores.div(masked_scores.sum(1, keepdim=True))
    centroid = (vectors * normed_scores).sum(1)
    return centroid, normed_scores


def orthogonalize(param):
    """
    https://github.com/kevinzakka/pytorch-goodies#orthogonal-regularization
    """
    param_flat = param.view(param.shape[0], -1)
    product = torch.mm(param_flat, torch.t(param_flat))
    loss = product - torch.eye(param_flat.shape[0], device=param_flat.device)
    return torch.sqrt(torch.pow(loss.mean(), 2))


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def param_grad_wrt_loss(optimizers, module, loss):
    loss.backward(retain_graph=True)

    grad_norms = {n: p.grad.norm().item() for n, p in
                  module.named_parameters()
                  if p.grad is not None}

    for optimizer in optimizers:
        optimizer.zero_grad()

    return grad_norms


def grad_wrt_loss(optimizers, module, loss, filter=None):
    loss.backward(retain_graph=True)

    grad_norms = [(n, p.grad.norm().item()) for n, p in
                  module.named_parameters()
                  if p.grad is not None]

    if filter is not None:
        grad_norms = [g for g in grad_norms if filter in g[0]]

    mean_norm = mean([gn for n, gn in grad_norms])

    for optimizer in optimizers:
        optimizer.zero_grad()

    return mean_norm


def _extractive_mask(tokens, _vocab):
    mask = torch.zeros(tokens.size(0), len(_vocab), device=tokens.device)
    mask = mask.scatter_(1, tokens, 1).byte()
    mask[:, _vocab.tok2id[_vocab.SOS]] = 0
    mask[:, _vocab.tok2id[_vocab.PAD]] = 0
    mask[:, _vocab.tok2id[_vocab.EOS]] = 1
    return mask


def fake_inputs(inputs, latent_lengths, sos, pad=1):
    batch_size, seq_len = inputs.size()

    if latent_lengths is not None:
        max_length = max(latent_lengths)
    else:
        max_length = seq_len + pad

    fakes = torch.zeros(batch_size, max_length, device=inputs.device)
    fakes = fakes.type_as(inputs)
    fakes[:, 0] = sos
    return fakes


def last_by_index(outputs, lengths):
    # Index of the last output for each sequence.
    idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                           outputs.size(2)).unsqueeze(1)
    return outputs.gather(1, idx).squeeze()


def last_timestep(outputs, lengths, bi=False):
    if bi:
        forward, backward = split_directions(outputs)
        last_forward = last_by_index(forward, lengths)
        last_backward = backward[:, 0, :]
        return torch.cat((last_forward, last_backward), dim=-1)

    else:
        return last_by_index(outputs, lengths)


def split_directions(outputs):
    direction_size = int(outputs.size(-1) / 2)
    forward = outputs[:, :, :direction_size]
    backward = outputs[:, :, direction_size:]
    return forward, backward


def transfer_weights(target, source):
    target_params = target.named_parameters()
    source_params = source.named_parameters()

    dict_target_params = dict(target_params)

    for name, param in source_params:
        if name in dict_target_params:
            dict_target_params[name].data.copy_(param.data)


def tie_weights(target, source):
    target_params = target.named_parameters()
    source_params = source.named_parameters()

    dict_target_params = dict(target_params)

    for name, param in source_params:
        if name in dict_target_params:
            setattr(target, name, getattr(source, name))


def length_countdown(lengths):
    batch_size = lengths.size(0)
    max_length = max(lengths)
    desired_lengths = lengths - 1

    _range = torch.arange(0, -max_length, -1, device=lengths.device)
    _range = _range.repeat(batch_size, 1)
    _countdown = _range + desired_lengths.unsqueeze(-1)

    return _countdown


def drop_tokens(embeddings, word_dropout):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - word_dropout)
    embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    return embeddings, mask


def lm_fusion(logits_tm, logits_lm, fusion, a=None):
    """
    POSTNORM: Combine with the LM *after* normalizing the TM
    PRENORM: Combine with the LM *before* normalizing the TM

    :param logits_tm: The logits of the Translation Model
    :param logits_lm: The logits of the Language Model
    :param fusion: The fusion method to use from combining the logits
    :param a: Interpolation weight - applicable only to shallow-fusion
    :return: logits
    """
    if fusion == "postnorm":
        logits = F.log_softmax(logits_tm, -1) + F.log_softmax(logits_lm, -1)

    elif fusion == "prenorm":
        logits = logits_tm + F.log_softmax(logits_lm, -1)

    elif fusion == "shallow":
        assert 0 < a < 1
        logp_tm = F.log_softmax(logits_tm, -1)
        logp_lm = F.log_softmax(logits_lm, -1)
        logits = (1 - a) * logp_tm + a * logp_lm
    else:
        raise ValueError

    return logits
