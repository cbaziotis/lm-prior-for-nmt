import html
import os
import random

import torch
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

from helpers.text import devectorize, detokenize
from modules.callbacks import TrainerCallback


class SentLMSamplesCallback(TrainerCallback):
    def __init__(self, interval, max_len=50, n_samples=10, **kwargs):
        super().__init__(interval, **kwargs)
        self.max_len = max_len
        self.n_samples = n_samples

    def _get_inputs(self, t):
        _dev = next(t.model.parameters()).device
        dataset = t.valid_loader.dataset
        ids = random.sample(range(0, len(dataset.data)), 2 * self.n_samples)
        b = [dataset.dataitem(i) for i in ids]
        b = [x for x in b if 5 < len(x) < self.max_len][:self.n_samples]
        lens = torch.tensor([len(x) for x in b], dtype=torch.long, device=_dev)
        inputs = [torch.tensor(x, dtype=torch.long, device=_dev) for x in b]
        inputs = pad_sequence(inputs, True)
        return inputs, lens

    def _predict(self, t, temp=1, sos_id=1):
        tokens = []
        hidden = None

        device = next(t.model.parameters()).device
        sos_ids = torch.zeros(self.n_samples, 1, device=device,
                              dtype=torch.long)
        sos_ids.fill_(sos_id)
        lengths = torch.ones(self.n_samples, device=device, dtype=torch.long)
        tokens.append(sos_ids)

        for i in range(self.max_len):
            if t.config["model"]["type"] == "rnn":
                x_i = t.model(tokens[-1], lengths, hidden)
            elif t.config["model"]["type"] == "transformer":
                x_i = t.model(torch.cat(tokens, 1), lengths * len(tokens))
                x_i["logits"] = x_i["logits"][:, -1, :].unsqueeze(1)
            else:
                raise NotImplementedError

            if temp == 0:
                if i == 0 and t.config["model"]["mode"] == "LM":
                    dist = torch.distributions.Multinomial(logits=x_i["logits"])
                    samples = dist.sample().max(-1)[1]
                else:
                    samples = x_i["logits"].max(-1)[1]
            else:
                dist = torch.distributions.Multinomial(logits=x_i["logits"]
                                                       .div(temp))
                samples = dist.sample().max(-1)[1]

            tokens.append(samples)
            if t.config["model"]["type"] == "rnn":
                hidden = x_i["hidden"]

        return torch.cat(tokens[1:], -1).tolist()

    @staticmethod
    def norm(x):
        return x.replace('<', '≺').replace('>', '≻')

    def tok2text(self, inputs, preds, vocab):
        _t2i = lambda x: devectorize(x,
                                     vocab.id2tok, vocab.tok2id[vocab.EOS],
                                     strip_eos=True)
        bpe = vocab.subword is not None
        if inputs is None:
            row = lambda x: f"<p>{self.norm(x)}</p>"
            samples = [row(html.unescape(detokenize(y, bpe)))
                       for y in _t2i(preds)]
        else:
            row = lambda x, y: f"<p>INP: {self.norm(x)} <br/>" \
                               f"REC: {self.norm(y)}</p>"
            src = [html.unescape(detokenize(x[1:], bpe))
                   for x in _t2i(inputs.tolist())]
            y_toks = [html.unescape(detokenize(x, bpe))
                      for x in _t2i(preds)]
            samples = [row(x, y) for x, y in zip(src, y_toks)]
        return ''.join(samples)

    def _get_samples(self, trainer):
        inputs, lens, hidden = None, None, None
        text = ""
        vocab = trainer.get_vocab()
        for temp in [1, 0.5, 0.3, 0]:
            preds = self._predict(trainer, temp, vocab.SOS_id)
            samples = self.tok2text(inputs, preds, vocab)
            text += f"<p><strong>Temperature:{temp}</strong>{samples}</p>"

        del inputs, hidden, lens
        return text

    def batch_end(self, t, batch, epoch_losses, batch_losses, batch_outputs):
        # skip
        if t.step % self.interval != 0:
            return

        try:
            t.model.eval()
            with torch.no_grad():
                samples_log = self._get_samples(t)
            html_samples = f'<style> body, p {{ font-family: "Dejavu Sans Mono", ' \
                           f'serif; font-size: 12px; }}</style>{samples_log}'

            t.exp.text("samples", html_samples, "Samples")

            with open(os.path.join(t.exp.output_dir, "samples.html"),
                      "w") as f:
                f.write(html_samples)

        except Exception as e:
            print("The following error occurred in LMSamplesCallback:")
            print(e)

        t.model.train()
