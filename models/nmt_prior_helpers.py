"""
When transferring weights from a pretrained model,
we want to override ONLY those hyper-parameters that are relevant
for the correct transfer of weights.

e.g. num of layers, size of layers, structure

All other hyper-parameters of the target model should be left unchanged!
"""
import glob
import json
import math
import os
import pickle

import numpy
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.training import load_checkpoint
from models.nmt_prior_trainer import NmtPriorTrainer
from models.translate import eval_nmt_checkpoint, seq2seq_translate
from modules.data.collates import Seq2SeqCollate
from modules.data.datasets import SequenceDataset, TranslationDataset
from modules.data.loaders import MultiDataLoader
from modules.data.samplers import BucketTokensSampler
from modules.data.utils import fix_paths
from modules.models import RNNLM, TransformerLM
from sys_config import TRAINED_PATH


def nmt_dataloaders(config, src_vocab=None, trg_vocab=None):
    """
    Isolate the whole process in this function, in order to reduce the
    cognitive load of reading the main script

    Notes: All the datasets/dataloaders operating on the same language
    must use the same preprocessing, vocabulary etc.

    Args:
        config:
        preprocess:
        src_vocab:
        trg_vocab:

    Returns:

    """
    b_toks = config["batch_tokens"]

    print("Building training dataset...")

    train_src = SequenceDataset(config["data"]["src"]["train_path"],
                                vocab=src_vocab,
                                **{**config["data"], **config["data"]["src"]})
    train_trg = SequenceDataset(config["data"]["trg"]["train_path"],
                                vocab=trg_vocab,
                                **{**config["data"], **config["data"]["trg"]})
    train_set = TranslationDataset(train_src, train_trg)

    print("Building validation dataset...")
    val_src = SequenceDataset(config["data"]["src"]["val_path"],
                              vocab=train_src.vocab,
                              **{**config["data"], **{"subsample": 0},
                                 **config["data"]["src"]})
    val_trg = SequenceDataset(config["data"]["trg"]["val_path"],
                              vocab=train_trg.vocab,
                              **{**config["data"], **{"subsample": 0},
                                 **config["data"]["trg"]})
    val_set = TranslationDataset(val_src, val_trg)

    train_lengths = train_src.lengths + train_trg.lengths
    val_lengths = val_src.lengths + val_trg.lengths

    train_loader = DataLoader(
        train_set, num_workers=config["cores"], pin_memory=config["pin_memory"],
        batch_sampler=BucketTokensSampler(train_lengths, b_toks, shuffle=True),
        collate_fn=Seq2SeqCollate())

    val_loader = DataLoader(
        val_set, num_workers=config["cores"], pin_memory=config["pin_memory"],
        batch_sampler=BucketTokensSampler(val_lengths, b_toks),
        collate_fn=Seq2SeqCollate())

    if "synthetic" in config["data"]:
        bt_src = SequenceDataset(config["data"]["synthetic"]["src_path"],
                                 vocab=train_src.vocab,
                                 **{**config["data"], **{"subsample": 0},
                                    **config["data"]["src"]})
        bt_trg = SequenceDataset(config["data"]["synthetic"]["trg_path"],
                                 vocab=train_trg.vocab,
                                 **{**config["data"], **{"subsample": 0},
                                    **config["data"]["trg"]})
        bt_set = TranslationDataset(bt_src, bt_trg)

        bt_lengths = bt_src.lengths + bt_trg.lengths
        bt_loader = DataLoader(bt_set,
                               num_workers=config["cores"],
                               pin_memory=config["pin_memory"],
                               batch_sampler=BucketTokensSampler(bt_lengths,
                                                                 b_toks,
                                                                 shuffle=True),
                               collate_fn=Seq2SeqCollate())

        train_loader = MultiDataLoader([train_loader, bt_loader], "cycle",
                                       names=["gold", "synthetic"])

    return train_loader, val_loader


def eval_best(trainer: NmtPriorTrainer):
    # compute the entropy per token on the val and test sets
    try:
        eval_entropy(trainer, 'val_path')
        eval_entropy(trainer, 'test_path')
    except:
        pass

    # compute the model's BLEU score on the val and test sets
    beams = [1, 5]
    eval_nmt_checkpoint(trainer.best_checkpoint,
                        trainer.config["device"], beams=beams)

    if trainer.config["data"]["prior_path"] is None:

        lang = trainer.config["data"]["trg"]["lang"]
        _model = trainer.config["model"].get("type", "rnn")
        if _model == "transformer":
            _model = "trans"

        for lm in glob.glob(os.path.join(TRAINED_PATH,
                                         f"prior.lm_news_{lang}*{_model}*.pt")):
            for fusion_a in [0.1]:
                eval_nmt_checkpoint(trainer.best_checkpoint,
                                    trainer.config["device"],
                                    beams=beams, lm=lm, fusion_a=fusion_a)


def backtranslate(trainer: NmtPriorTrainer):
    cp = load_checkpoint(trainer.best_checkpoint)
    fusion = trainer.config["model"]["decoding"].get("fusion")
    src_file = fix_paths(trainer.config["data"]["backtranslate_path"])

    _base, _file = os.path.split(src_file)
    out_file = os.path.join(_base, f"{trainer.config['name']}.synthetic")

    # if trainer.config['resume_state_id'] is not None:
    #     out_file += "__" + trainer.config['resume_state_id']

    seq2seq_translate(checkpoint=cp,
                      src_file=src_file,
                      out_file=out_file,
                      beam_size=1,
                      length_penalty=1,
                      lm=None,
                      fusion=fusion,
                      fusion_a=None,
                      batch_tokens=trainer.config["batch_tokens"],
                      device=trainer.device)


def eval_entropy(trainer: NmtPriorTrainer, dataset):
    """
    Calculate the per-timestep entropy of a model on a given test set
    :param trainer:
    :param dataset: either val_path or test_path
    :return:
    """
    src_vocab = trainer.valid_loader.dataset.src.vocab
    trg_vocab = trainer.valid_loader.dataset.trg.vocab
    val_src = SequenceDataset(trainer.config["data"]["src"][dataset],
                              vocab=src_vocab,
                              **{**trainer.config["data"], **{"subsample": 0},
                                 **trainer.config["data"]["src"]})
    val_trg = SequenceDataset(trainer.config["data"]["trg"][dataset],
                              vocab=trg_vocab,
                              **{**trainer.config["data"], **{"subsample": 0},
                                 **trainer.config["data"]["trg"]})
    val_set = TranslationDataset(val_src, val_trg)

    sampler = BucketTokensSampler(val_src.lengths + val_trg.lengths,
                                  trainer.config["batch_tokens"])
    data_loader = DataLoader(val_set, batch_sampler=sampler,
                             collate_fn=Seq2SeqCollate())

    # ----------------------------------------------------------------
    flatten = lambda l: [item for sublist in l for item in sublist]

    def _entropy(logits, length):
        h = Categorical(logits=logits).entropy().tolist()
        return flatten([x[:l] for x, l in zip(h, length)])

    # ----------------------------------------------------------------
    losses = []

    entropies = {"lm": [], "dec": [], "tm": []}

    for batch in tqdm(data_loader, total=len(data_loader),
                      desc="Translating..."):
        batch = trainer.batch_to_device(batch)
        x_sos, x_eos, x_len, y_sos, y_eos, y_len = batch
        batch_losses, batch_outputs = trainer.process_batch(*batch)

        # save CE loss
        losses.append(batch_losses["mt"].item())

        # save entropies per timestep
        dec = batch_outputs['model_outputs'][1]
        entropies["tm"].extend(_entropy(dec["logits"], y_len))

        if "dec" in dec and dec["dec"] is not None and len(dec["dec"]) > 0:
            entropies["dec"].extend(_entropy(dec["dec"], y_len))

        if "lm" in dec and dec["lm"] is not None and len(dec["lm"]) > 0:
            entropies["lm"].extend(_entropy(dec["lm"], y_len))
        elif trainer.prior is not None:
            lm_logits = trainer.prior(y_sos, y_len)["logits"]
            entropies["lm"].extend(_entropy(lm_logits, y_len))

        if batch_outputs is not None:
            trainer.empty_batch_outputs(batch_outputs)
            batch_outputs.clear()
        batch_losses.clear()
        del batch[:]
        del batch_losses, batch_outputs, batch

    base_dir = trainer.exp.output_dir
    suffix = dataset.split('_')[0]

    with open(os.path.join(base_dir, f"ppl_{suffix}.json"), "w") as f:
        loss = {"ce": numpy.mean(losses), "ppl": math.exp(numpy.mean(losses))}
        json.dump(loss, f, indent=4)

    with open(os.path.join(base_dir, f"entropies_{suffix}.pkl"), 'wb') as f:
        pickle.dump(entropies, f)

    with open(os.path.join(base_dir, f"entropies_{suffix}.json"), "w") as f:
        json.dump(entropies, f, indent=4)


def prior_model_from_checkpoint(cp):
    model_type = cp["config"]["model"].get("type", "rnn")

    if model_type == "rnn":
        prior_model = RNNLM
    elif model_type == "transformer":
        prior_model = TransformerLM
    else:
        raise NotImplementedError

    prior = prior_model(len(cp['vocab']), **cp["config"]["model"])
    prior.load_state_dict(cp["model"])

    # due to a bug in PyTorch we cannot backpropagate through a model in eval
    # mode. Therefore, we have to manually turn off the regularizations.
    for name, module in prior.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

        elif isinstance(module, nn.LSTM):
            module.dropout = 0

        elif isinstance(module, nn.GRU):
            module.dropout = 0

    return prior
