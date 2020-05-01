import math
import math
import os

import numpy
import pandas
import torch

from helpers.eval import get_bleu_score
from helpers.text import devectorize
from helpers.viz import seq2seq_attentions
from models.translate import seq2seq_translate_ids, \
    seq2seq_output_ids_to_file
from modules.callbacks import TrainerCallback
from modules.trainer import Trainer
from mylogger.viz_samples import samples2html


class EvalCallback(TrainerCallback):
    def __init__(self, interval,
                 keep_best=False,
                 early_stop=10,
                 **kwargs):
        super().__init__(interval, **kwargs)
        self.early_stop = early_stop
        self.patience = early_stop
        self.keep_best = keep_best
        self.best = None

    def _score(self, t):
        # t.free_gpu()
        t.model.eval()

        trg_vocab = t.get_vocab()[1]
        hyp_path = os.path.join(t.exp.output_dir, f"hyps.txt")
        val_path = t.config["data"]["trg"]["val_path"]

        fusion = t.config["model"]["decoding"].get("fusion")
        decoding = {}
        if fusion is not None:
            decoding = {"fusion": fusion, "lm": t.prior}

        output_ids = seq2seq_translate_ids(t.model, t.valid_loader, trg_vocab,
                                           **decoding)
        output_ids = output_ids[t.valid_loader.batch_sampler.reverse_ids]
        seq2seq_output_ids_to_file(output_ids, trg_vocab, hyp_path)
        bleu = get_bleu_score(hyp_path, val_path)

        t.model.train()

        return bleu

    def batch_start(self, t: Trainer):
        # skip
        if t.step % self.interval != 0:
            return

        with torch.no_grad():
            bleu = self._score(t)
            print(f"BLEU:{bleu}\n")

            if self.best is None or bleu > self.best:
                self.best = bleu
                self.patience = self.early_stop
                if self.keep_best:
                    t.checkpoint(name=t.config["name"], tags=["best"])

                # save the best perplexity and bleu score
                val_loss = t.eval_epoch(only_eval=True)
                ce_loss = pandas.DataFrame(val_loss)["mt"].mean()
                text = f"BLEU:{bleu}" \
                       f"\nCross-Entropy:{ce_loss:.2f}" \
                       f"\nPerplexity:{math.exp(ce_loss):.2f}"
                t.exp.text("best_scores", text, "Best scores")

            else:
                self.patience -= 1

                if self.patience < 0:
                    t.early_stop = True

            t.exp.line("bleu", None, "BLEU", bleu)


class AttentionCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_backward_end(self, t, batch, epoch_losses, batch_losses,
                           batch_outputs):
        # skip
        if t.step % self.interval != 0:
            return

        enc, dec = batch_outputs['model_outputs']
        src_vocab = t.valid_loader.dataset.src.vocab
        trg_vocab = t.valid_loader.dataset.trg.vocab
        devec = lambda t, v: devectorize(t.tolist(), v.id2tok, v.tok2id[v.EOS])
        src = devec(batch[1], src_vocab)
        hyp = devec(dec["logits"].max(dim=2)[1], trg_vocab)

        file = os.path.join(t.exp.output_dir, "attentions.pdf")
        seq2seq_attentions(src[:5], hyp[:5], dec["attention"][:5].tolist(),
                           file)


class SamplesCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _batch_backward_end(self, t, batch, epoch_losses, batch_losses,
                            batch_outputs):

        # skip
        if t.step % self.interval != 0:
            return

        enc, dec = batch_outputs['model_outputs']
        src_vocab = t.valid_loader.dataset.src.vocab
        trg_vocab = t.valid_loader.dataset.trg.vocab

        devec = lambda t, v: devectorize(t.tolist(), v.id2tok, v.tok2id[v.EOS],
                                         True)
        src = devec(batch[1], src_vocab)
        hyp = devec(dec["logits"].max(dim=2)[1], trg_vocab)
        trg = devec(batch[4], trg_vocab)

        if t.config["model"]["decoder"]["learn_tau"]:
            tau = numpy.around(numpy.array(dec["tau"].tolist()), 2)
            t0 = t.config["model"]["decoder"]["tau_0"]
            tau_bound = round(1 / (math.log(1 + math.exp(-100)) + t0), 2)
            tau_opacity = (tau / tau_bound).tolist()
            tau = tau.tolist()
        else:
            tau = None

        samples = []
        for i in range(len(src)):
            sample = []

            _src = {"tag": "SRC", "tokens": src[i], "color": "0, 0, 0"}
            sample.append(_src)

            _hyp = {"tag": "HYP", "tokens": hyp[i], "color": "0, 0, 0"}
            sample.append(_hyp)

            if t.config["model"]["decoder"]["learn_tau"]:
                _tau = {"tag": "TAU", "tokens": list(map(str, tau[i])),
                        "opacity": tau_opacity[i], "color": "255, 100, 100",
                        "normalize": False}
                sample.insert(2, _tau)

            _trg = {"tag": "TRG", "tokens": trg[i], "color": "0, 0, 0"}
            sample.append(_trg)

            samples.append(sample)

        html_samples = samples2html(samples)
        t.exp.text("samples", html_samples, "Samples", pre=False)

    @staticmethod
    def norm(x):
        return x.replace('<', '≺').replace('>', '≻')

    def batch_backward_end(self, t, batch, epoch_losses, batch_losses,
                           batch_outputs):

        # skip
        if t.step % self.interval != 0:
            return

        enc, dec = batch_outputs['model_outputs']
        src_vocab = t.valid_loader.dataset.src.vocab
        trg_vocab = t.valid_loader.dataset.trg.vocab
        row = lambda x, z, y: f"<p>INP: {self.norm(x)} <br/>" \
                              f"HYP: {self.norm(z)} <br/>" \
                              f"TRG: {self.norm(y)}</p>"

        devec = lambda t, v: devectorize(t.tolist(), v.id2tok, v.tok2id[v.EOS],
                                         True)
        src = devec(batch[1], src_vocab)
        hyp = devec(dec["logits"].max(dim=2)[1], trg_vocab)
        trg = devec(batch[4], trg_vocab)

        src = [src_vocab.detokenize(x) for x in src]
        hyp = [trg_vocab.detokenize(x) for x in hyp]
        trg = [trg_vocab.detokenize(x) for x in trg]

        samples = [row(x, z, y) for x, z, y in zip(src, hyp, trg)]

        html_samples = f'<style> body, p {{ font-family: "Dejavu Sans Mono", ' \
                       f'serif; font-size: 12px; }}</style>{"".join(samples)}'

        t.exp.text("samples", html_samples, "Samples")

        with open(os.path.join(t.exp.output_dir, "samples.html"), "w") as f:
            f.write(html_samples)
