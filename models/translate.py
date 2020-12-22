import argparse
import json
import os
import subprocess
import sys

import numpy
import pandas
# sys.path.insert(0, '.')
# sys.path.insert(0, '..')
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.text import devectorize
from helpers.training import load_checkpoint
from modules.data.collates import LMCollate
from modules.data.datasets import SequenceDataset
from modules.data.samplers import BucketTokensSampler
from modules.data.utils import fix_paths
from modules.models import Seq2SeqTransformer, Seq2SeqRNN, RNNLM, TransformerLM


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


def seq2seq_translate_ids(model,
                          data_loader,
                          vocab,
                          **kwargs):
    """
    Translate greedily the data in the data_loader and return the token ids
    """
    output_ids = []
    device = next(model.parameters()).device
    sos_id = vocab.SOS_id
    eos_id = vocab.EOS_id
    pad_id = vocab.PAD_id

    beam = kwargs.get("beam_size", 1)

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader),
                          desc=f"Translating (beam={beam})..."):
            batch = list(map(lambda x: x.to(device), batch))

            if beam == 1:
                _, dec = model.translate(batch[0], batch[2], sos_id, **kwargs)
                output_ids.extend(dec["logits"].max(2)[1].tolist())
                dec.clear()
                del batch[:]
                del batch, dec

            else:
                batch_ids = model.beam(batch[0], batch[2],
                                       sos_id, eos_id, pad_id, **kwargs)
                output_ids.extend(batch_ids)

    return numpy.array(output_ids)


def seq2seq_output_ids_to_file(output_ids, trg_vocab, out_file):
    """
    Devectorize and Detokenize the translated token ids and write the
    translations to a text file
    """
    output_tokens = devectorize(output_ids.tolist(),
                                trg_vocab.id2tok,
                                trg_vocab.EOS_id,
                                strip_eos=True,
                                pp=True)

    with open(out_file, "w") as fo:
        for sent in output_tokens:
            text = trg_vocab.detokenize(sent)
            fo.write(text + "\n")


def eval_nmt_checkpoint(checkpoint, device, beams=None, lm=None, fusion_a=None,
                        results=None, results_low=None):
    if beams is None:
        beams = [1, 5, 10]

    _base, _file = os.path.split(checkpoint)
    cp = load_checkpoint(checkpoint)

    def score(dataset, beam_size) -> (float, float):
        hyp_file = os.path.join(_base, f"hyps_{dataset}_beam-{beam_size}.txt")
        src_file = cp["config"]["data"]["src"][f"{dataset}_path"]
        ref_file = cp["config"]["data"]["trg"][f"{dataset}_path"]

        src_file = fix_paths(src_file, "datasets")
        ref_file = fix_paths(ref_file, "datasets")

        fusion = cp["config"]["model"]["decoding"].get("fusion")
        batch_tokens = max(10000 // beam_size, 1000)

        if fusion is None and lm is not None and fusion_a is not None:
            fusion = "shallow"

        seq2seq_translate(checkpoint=cp,
                          src_file=src_file,
                          out_file=hyp_file,
                          beam_size=beam_size,
                          length_penalty=1,
                          lm=lm,
                          fusion=fusion,
                          fusion_a=fusion_a,
                          batch_tokens=batch_tokens,
                          device=device)
        _mixed = compute_bleu_score(hyp_file, ref_file)
        _lower = compute_bleu_score(hyp_file, ref_file, True)
        return _mixed, _lower

    if results is None:
        results = {d: {k: None for k in beams} for d in ["val", "test"]}
    if results_low is None:
        results_low = {d: {k: None for k in beams} for d in ["val", "test"]}

    for d in ["val", "test"]:
        for k in beams:
            try:
                mixed, lower = score(d, k)
                results[d][k] = mixed
                results_low[d][k] = lower
            except Exception as e:
                print(e)
                results[d][k] = None
                results_low[d][k] = None

    text = pandas.DataFrame.from_dict(results).to_string()
    name = "BLEU"
    if fusion_a is not None:
        name += f"_shallow_{fusion_a}_{lm.split('.')[-2]}"
    with open(os.path.join(_base, f"{name}.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(_base, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=4)


def seq2seq_translate(checkpoint,
                      src_file,
                      out_file,
                      beam_size,
                      length_penalty,
                      lm,
                      fusion,
                      fusion_a,
                      batch_tokens,
                      device):
    # --------------------------------------
    # load checkpoint
    # --------------------------------------
    if isinstance(checkpoint, str):
        cp = load_checkpoint(checkpoint)
    else:
        cp = checkpoint
    src_vocab, trg_vocab = cp["vocab"]

    # --------------------------------------
    # load model
    # --------------------------------------
    model_type = cp["config"]["model"].get("type", "rnn")
    src_ntokens = len(src_vocab)
    trg_ntokens = len(trg_vocab)

    if model_type == "rnn":
        model = Seq2SeqRNN(src_ntokens, trg_ntokens, **cp["config"]["model"])
    elif model_type == "transformer":
        model = Seq2SeqTransformer(src_ntokens, trg_ntokens,
                                   **cp["config"]["model"])
    else:
        raise NotImplementedError

    model.load_state_dict(cp["model"])
    model.to(device)
    model.eval()

    # --------------------------------------
    # load prior
    # --------------------------------------
    if lm is not None:
        lm_cp = load_checkpoint(lm)
    elif fusion:
        lm_cp = load_checkpoint(fix_paths(cp["config"]["data"]["prior_path"],
                                          "checkpoints"))
    else:
        lm_cp = None

    if lm_cp is not None:
        lm = prior_model_from_checkpoint(lm_cp)
        lm.to(device)
        lm.eval()
    else:
        lm = None

    test_set = SequenceDataset(src_file,
                               vocab=src_vocab,
                               **{**cp["config"]["data"], **{"subsample": 0},
                                  **cp["config"]["data"]["src"]})
    print(test_set)

    if batch_tokens is None:
        batch_tokens = cp["config"]["batch_tokens"]

    sampler = BucketTokensSampler(test_set.lengths * 2, batch_tokens)
    data_loader = DataLoader(
        test_set,
        # num_workers=cp["config"].get("cores",
        #                              min(4, multiprocessing.cpu_count())),
        # pin_memory=cp["config"].get("pin_memory", True),
        num_workers=cp["config"].get("cores", 4),
        pin_memory=True,
        batch_sampler=sampler,
        collate_fn=LMCollate())

    # translate the data
    output_ids = seq2seq_translate_ids(model,
                                       data_loader,
                                       trg_vocab,
                                       beam_size=beam_size,
                                       length_penalty=length_penalty,
                                       lm=lm,
                                       fusion=fusion,
                                       fusion_a=fusion_a)

    output_ids = output_ids[data_loader.batch_sampler.reverse_ids]
    seq2seq_output_ids_to_file(output_ids, trg_vocab, out_file)


def compute_bleu_score(preds, ref, lc=False):
    score_proc = lambda x: subprocess.Popen(x, shell=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
    try:
        cmd = f"cat {preds} | sacrebleu {ref}  -b --force"
        if lc:
            cmd += " -lc"
        output, error = score_proc(cmd).communicate()
        bleu = float(output.strip())
    except:
        sacrebleu_path = sys.executable.replace("python", "sacrebleu")
        cmd = f"cat {preds} | {sacrebleu_path} {ref}  -b --force"
        if lc:
            cmd += " -lc"
        output, error = score_proc(cmd).communicate()
        bleu = float(output.strip())
    return bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        help="Preprocessed input file, in source language.")
    parser.add_argument('--out',
                        help="The name of the *detokenized* output file, in the target language")
    parser.add_argument('--cp', help="The checkpoint of the translation model.")
    parser.add_argument('--ref',
                        help="The raw reference file, for internally compute BLEU with sacreBLEU.")
    parser.add_argument('--beam_size', type=int,
                        help="The width of the beam search (default=1)")
    parser.add_argument('--length_penalty', default=1.0, type=float,
                        help="The value of the length penalty (default=1.0)")
    parser.add_argument('--lm',
                        help="The checkpoint of a pretrained language model."
                             "Applicable when using 1) LM fusion, or 2) LM prior.")
    parser.add_argument('--fusion',
                        help="The type of LM-fusion to use. "
                             "Options: [shallow, postnorm, prenorm]")
    parser.add_argument('--fusion_a',
                        help="This is the weight for the LM in shallow-fusion.",
                        type=float)
    parser.add_argument('--batch_tokens',
                        help="The size of the batch in number of tokens.",
                        default=4000, type=int)
    parser.add_argument('--device',
                        help="The devide id to use "
                             "(e.g., cuda, cuda:2, cpu, ...)",
                        default="cuda")

    args = parser.parse_args()

    print(args)

    seq2seq_translate(checkpoint=args.cp,
                      src_file=args.src,
                      out_file=args.out,
                      beam_size=args.beam_size,
                      length_penalty=args.length_penalty,
                      lm=args.lm,
                      fusion=args.fusion,
                      fusion_a=args.fusion_a,
                      batch_tokens=args.batch_tokens,
                      device=args.device)

    if args.ref is not None:
        bleu = compute_bleu_score(args.out, args.ref)
        print(f"BLEU:{bleu}")

    # # --------------------------------------------------------------------
    # # --------------------------------------------------------------------
    # # seq2seq.proto_entr_deep
    # checkpoint = "../experiments/trans.deen_base/20-12-21_21:41:08/trans.deen_base_best.pt"
    # # lm = "../checkpoints/prior.lm_news_en_best.pt"
    # lm = None
    # src_file = "../datasets/mt/wmt_ende/test.de.pp"
    # out_file = "../datasets/mt/wmt_ende/test.en.pp.hyps"
    # ref_file = "../datasets/mt/wmt_ende/test.en.pp"
    # beam_size = 2
    # length_penalty = 1
    # # fusion = "shallow"
    # fusion = None
    # fusion_a = 0.0
    # batch_tokens = 2000
    # device = "cpu"
    #
    # seq2seq_translate(checkpoint=checkpoint,
    #                   src_file=src_file,
    #                   out_file=out_file,
    #                   beam_size=beam_size,
    #                   length_penalty=length_penalty,
    #                   lm=lm,
    #                   fusion=fusion,
    #                   fusion_a=fusion_a,
    #                   batch_tokens=batch_tokens, device=device)
    # bleu = get_bleu_score(out_file, ref_file)
    # print(f"BLEU:{bleu}")
