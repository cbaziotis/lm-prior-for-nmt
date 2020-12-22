import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.text import devectorize
from helpers.training import load_checkpoint
from models.translate import prior_model_from_checkpoint
from modules.data.collates import Seq2SeqCollate
from modules.data.datasets import SequenceDataset, TranslationDataset
from modules.helpers import sequence_mask
from modules.models import Seq2SeqTransformer


def _ce_loss(logits, labels, lengths, ignore_index=0):
    _logits = logits.contiguous().view(-1, logits.size(-1))

    if ignore_index >= 0:
        _labels = labels.contiguous().view(-1)
    else:
        assert lengths is not None
        mask = ~sequence_mask(lengths, labels.size(1))
        _labels = labels.masked_fill_(mask, -1).contiguous().view(-1)

    _loss = F.cross_entropy(_logits, _labels, ignore_index=ignore_index,
                            reduction='none')
    _loss_per_step = _loss.view(labels.size())
    loss = _loss_per_step.sum(-1) / lengths.float()
    return loss, _loss_per_step


def _load_model(checkpoint, device="cuda"):
    cp = load_checkpoint(checkpoint)
    x_vocab, y_vocab = cp["vocab"]
    model = Seq2SeqTransformer(len(x_vocab), len(y_vocab),
                               **cp["config"]["model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model.eval()
    return model, x_vocab, y_vocab, cp["config"]


def _logits2dist(logits, vocab, topk=5):
    top_p, top_i = torch.softmax(logits, -1).topk(topk, -1)
    t = []
    for probs, tokids in zip(top_p.tolist(), top_i.tolist()):
        t.append([(vocab.id2tok[t], p) for p, t in zip(probs, tokids)])
    return t


device = "cuda"
# --------------------------------------
# load model
# --------------------------------------
model_prior, _, _, _ = _load_model("final.trans.deen_prior_3M_kl_best.pt")
model_postnorm, _, _, _ = _load_model("final.trans.deen_postnorm_best.pt")
model_base, src_vocab, trg_vocab, cnf = _load_model("final.trans.deen_base_best.pt")
# --------------------------------------
# load lm
# --------------------------------------
lm_cp = "../checkpoints/prior.lm_news_en_30M_trans_best.pt"
lm_cp = load_checkpoint(lm_cp)
lm = prior_model_from_checkpoint(lm_cp)
lm.to(device)
lm.eval()

# --------------------------------------
# dataset
# --------------------------------------
src_path = "de.txt"
trg_path = "en.txt"
val_src = SequenceDataset(src_path, vocab=src_vocab, **{**cnf["data"],
                                                        **cnf["data"]["src"]})
val_trg = SequenceDataset(trg_path, vocab=trg_vocab, **{**cnf["data"],
                                                        **cnf["data"]["trg"]})
val_set = TranslationDataset(val_src, val_trg)
data_loader = DataLoader(
    val_set,
    batch_size=32,
    # batch_sampler=TokenBatchSampler(val_src.lengths * 2, 1000),
    collate_fn=Seq2SeqCollate())

results = []


def _is_failed(gold, y_ids, lm_ids, tm_ids):
    return [True if (g != y and g == t) else False
            for g, y, l, t in zip(gold, y_ids, lm_ids, tm_ids)]


def _batch_forward(batch):
    batch = list(map(lambda x: x.to(device), batch))
    x_sos, x_eos, x_len, y_sos, y_eos, y_len = batch

    # prior
    _, dec_prior = model_prior(x_eos, y_sos, x_len, y_len)
    dec_prior["lm"] = lm(y_sos, y_len)["logits"]

    # shallow
    _, dec_shallow = model_base(x_eos, y_sos, x_len, y_len,
                                **{"fusion": "shallow",
                                   "fusion_a": 0.1, "lm": lm})
    # postnorm
    _, dec_postnorm = model_postnorm(x_eos, y_sos, x_len, y_len,
                                     **{"fusion": "postnorm", "lm": lm})

    # --------------------------------------------------------------------
    _inputs = devectorize(x_eos.tolist(), src_vocab.id2tok, src_vocab.EOS_id,
                          strip_eos=True)
    _targets = devectorize(y_eos.tolist(), trg_vocab.id2tok, trg_vocab.EOS_id,
                           strip_eos=True)

    # --------------------------------------------------------------------
    # prior
    # --------------------------------------------------------------------
    _prior_ids = dec_prior["logits"].max(2)[1].tolist()
    _prior_lm_ids = dec_prior["lm"].max(2)[1].tolist()

    _prior_tokens = devectorize(_prior_ids, trg_vocab.id2tok)
    _prior_lm_tokens = devectorize(_prior_lm_ids, trg_vocab.id2tok)

    # --------------------------------------------------------------------
    # shallow
    # --------------------------------------------------------------------
    _shallow_ids = dec_shallow["logits"].max(2)[1].tolist()
    _shallow_tm_ids = dec_shallow["dec"].max(2)[1].tolist()
    _shallow_lm_ids = dec_shallow["lm"].max(2)[1].tolist()

    _shallow_fails = [_is_failed(y_eos[i], _shallow_ids[i],
                                 _shallow_lm_ids[i], _shallow_tm_ids[i])
                      for i in range(x_eos.size(0))]

    _shallow_tokens = devectorize(_shallow_ids, trg_vocab.id2tok)
    _shallow_tm_tokens = devectorize(_shallow_tm_ids, trg_vocab.id2tok)
    _shallow_lm_tokens = devectorize(_shallow_lm_ids, trg_vocab.id2tok)

    # --------------------------------------------------------------------
    # postnorm
    # --------------------------------------------------------------------
    _postnorm_ids = dec_postnorm["logits"].max(2)[1].tolist()
    _postnorm_tm_ids = dec_postnorm["dec"].max(2)[1].tolist()
    _postnorm_lm_ids = dec_postnorm["lm"].max(2)[1].tolist()

    _postnorm_fails = [_is_failed(y_eos[i], _postnorm_ids[i],
                                  _postnorm_lm_ids[i], _postnorm_tm_ids[i])
                       for i in range(x_eos.size(0))]

    _postnorm_tokens = devectorize(_postnorm_ids, trg_vocab.id2tok)
    _postnorm_tm_tokens = devectorize(_postnorm_tm_ids, trg_vocab.id2tok)
    _postnorm_lm_tokens = devectorize(_postnorm_lm_ids, trg_vocab.id2tok)
    # --------------------------------------------------------------------

    for i in range(x_eos.size(0)):

        if y_len[i].item() > 20:
            continue

        row = {
            "source": _inputs[i][:x_len[i]],
            "target": _targets[i][:y_len[i]],

            "prior_toks": _prior_tokens[i][:y_len[i]],
            "prior_toks_lm": _prior_lm_tokens[i][:y_len[i]],
            "prior_dist": _logits2dist(dec_prior["logits"][i], trg_vocab)[
                          :y_len[i]],
            "prior_dist_lm": _logits2dist(dec_prior["lm"][i], trg_vocab)[
                             :y_len[i]],

            "postnorm_toks": _postnorm_tokens[i][:y_len[i]],
            "postnorm_toks_lm": _postnorm_lm_tokens[i][:y_len[i]],
            "postnorm_toks_tm": _postnorm_tm_tokens[i][:y_len[i]],
            "postnorm_dist": _logits2dist(dec_postnorm["logits"][i], trg_vocab)[
                             :y_len[i]],
            "postnorm_dist_tm": _logits2dist(dec_postnorm["dec"][i], trg_vocab)[
                                :y_len[i]],
            "postnorm_dist_lm": _logits2dist(dec_postnorm["lm"][i], trg_vocab)[
                                :y_len[i]],
            "postnorm_fails": _postnorm_fails[i],

            "shallow_toks": _shallow_tokens[i][:y_len[i]],
            "shallow_toks_lm": _shallow_lm_tokens[i][:y_len[i]],
            "shallow_toks_tm": _shallow_tm_tokens[i][:y_len[i]],
            "shallow_dist": _logits2dist(dec_shallow["logits"][i], trg_vocab)[
                            :y_len[i]],
            "shallow_dist_tm": _logits2dist(dec_shallow["dec"][i], trg_vocab)[
                               :y_len[i]],
            "shallow_dist_lm": _logits2dist(dec_shallow["lm"][i], trg_vocab)[
                               :y_len[i]],
            "shallow_fails": _shallow_fails[i],

        }
        if any(_postnorm_fails[i]) or any(_shallow_fails[i]):
            yield row

    del batch


_results = []
with torch.no_grad():
    for batch in tqdm(data_loader, total=len(data_loader),
                      desc="Translating..."):
        _results.extend(list(_batch_forward(batch)))

with open("samples.json", "w") as f:
    json.dump(_results, f)
