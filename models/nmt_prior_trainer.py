from torch.nn import functional as F

from modules.helpers import sequence_mask, relax_softmax
from modules.losses import masked_kld, masked_mse, SequenceLoss, \
    differentiable_samples, dot3D
from modules.trainer import Trainer


class NmtPriorTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prior = kwargs.get("prior", None)

        trg_vocab = self.get_vocab()[1]
        smoothing = self.config["losses"]["mt"].get("smoothing", 0)
        device = next(self.model.parameters()).device

        ignore_index = -1 if trg_vocab.is_gpt2 else trg_vocab.PAD_id
        self.criterion = SequenceLoss(len(trg_vocab),
                                      device,
                                      ignore_index,
                                      smoothing)

    def empty_batch_outputs(self, outputs):
        outputs['model_outputs'][0].clear()
        outputs['model_outputs'][1].clear()

    def process_batch(self, x_sos, x_eos, x_len, y_sos, y_eos, y_len, **kwargs):
        """
        The inputs will be the following, assuming this pair of sentences:

        x = ['<sos>', 'every', 'clever', 'cat', 'hates', 'every', 'dog', '<eos>']
        y = ['<sos>', 'κάθε', 'έξυπνη', 'γάτα', 'μισεί', 'κάθε', 'σκύλο', '<eos>']

        Args:
            x_sos: ['<sos>', 'every', 'clever', 'cat', 'hates', 'every', 'dog']
            x_eos: ['every', 'clever', 'cat', 'hates', 'every', 'dog', '<eos>']
            x_len: 7

            y_sos: ['<sos>', 'κάθε', 'έξυπνη', 'γάτα', 'μισεί', 'κάθε', 'σκύλο']
            y_eos: ['κάθε', 'έξυπνη', 'γάτα', 'μισεί', 'κάθε', 'σκύλο', '<eos>']
            y_len: 7

            Note:

                _sos will be the input to decoders
                _eos will be the input to encoders and target for decoders

        Returns:

        """
        decoding = dict(self.config["model"].get("decoding", {}))

        if decoding.get("fusion") is not None:
            decoding["lm"] = self.prior

        outputs = self.model(x_eos, y_sos, x_len, y_len, **decoding)
        losses = dict()
        is_gpt2 = self.get_vocab()[1].is_gpt2

        # Loss calculation
        losses["mt"] = self.criterion(outputs[1]["logits"], y_eos, y_len)[0]

        if "prior" in self.config["losses"] and self.prior is not None:
            f_reg = self.config["losses"]["prior"].get("objective", "kl")

            if f_reg == "mse":
                lm_logits = self.prior(y_sos, y_len)["logits"]
                prior_loss, prior_loss_i = masked_mse(outputs[1]["logits"],
                                                      lm_logits,
                                                      y_len)

            elif f_reg in ["kl", "rkl"]:
                _tau = self.config["losses"]["prior"]["tau"]

                if is_gpt2:
                    _mask = sequence_mask(y_len, y_sos.size(1)).float()
                    lm_logits = self.prior(y_sos, attention_mask=_mask)[0]
                else:
                    lm_logits = self.prior(y_sos, y_len)["logits"]

                if f_reg == "kl":  # KL(p_prior, p_model)
                    prior_loss, prior_loss_i = masked_kld(outputs[1]["logits"],
                                                          lm_logits,
                                                          y_len,
                                                          _tau)
                else:  # rkl: KL(p_model, p_prior)
                    prior_loss, prior_loss_i = masked_kld(lm_logits,
                                                          outputs[1]["logits"],
                                                          y_len,
                                                          _tau)

                # multiply with tau^2 to make loss tau invariant
                prior_loss = prior_loss * (_tau ** 2)

            elif self.config["losses"]["prior"].get("objective", "kl") == "ppl":
                prob_tm = relax_softmax(outputs[1]["logits"],
                                        tau=1, gumbel=False, hard=False)
                prior_inps = differentiable_samples(self.prior.encoder.embed,
                                                    prob_tm, 1)
                lm_logits = self.prior(prior_inps, y_len)["logits"]
                mask = sequence_mask(y_len).float()
                prior_log_probs = F.log_softmax(lm_logits, -1)
                loss_i = dot3D(prob_tm.contiguous(),
                               prior_log_probs.contiguous()) * mask

                cross_entropy = loss_i.sum() / mask.sum()
                prior_loss = - cross_entropy
            else:
                raise ValueError

            losses["prior"] = prior_loss

        return losses, {'model_outputs': outputs}

    def get_vocab(self):
        _vocab = (self.valid_loader.dataset.src.vocab,
                  self.valid_loader.dataset.trg.vocab)
        return _vocab
