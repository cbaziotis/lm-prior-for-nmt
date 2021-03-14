import logging
import math

import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import \
    LabelSmoothedCrossEntropyCriterion
from fairseq.logging import metrics
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, base_architecture

logger = logging.getLogger(__name__)


@register_model('transformer_mt_lm')
class TransformerTranslationModelWithLanguageModel(TransformerModel):

    def __init__(self, args, translator, lm):
        super().__init__(args, translator.encoder, translator.decoder)
        self.args = args

        # attach the LM to the model to delegate parallelism etc. to fairseq
        self.lm = lm

        # freeze pretrained model
        self.lm.eval()
        for param in self.lm.parameters():
            param.requires_grad = False

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--lm-checkpoint', metavar='DIR',
                            help='path to load checkpoint from pretrained LM.')

    @classmethod
    def build_model(cls, args, task):
        transformer_mt_lm(args)
        translator = TransformerModel.build_model(args, task)

        # Load checkpoint of pretrained LM
        lm = checkpoint_utils.load_model_ensemble([args.lm_checkpoint])[0][0]

        return cls(args, translator, lm)

    def state_dict(self):
        """
        Omit the parameters of the pretrained LM from the checkpoint
        """
        state = TransformerModel.state_dict(self)
        for k, v in list(state.items()):
            if "lm." in k:
                del state[k]
        return state

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Fairseq expects that the returned state_dict should contain weights
        for the pretrained LM as we made it part of the model.
        However, these weights are not stored in the checkpoint and for this
        reason we add them here by copying them from the state_dict of the LM.
        """
        super().upgrade_state_dict_named(state_dict, name)

        # Put the weights of the pretrained LM into the state_dict
        model_state = TransformerModel.state_dict(self)
        for k, v in list(model_state.items()):
            if "lm." in k:
                state_dict[k] = v


@register_model_architecture("transformer_mt_lm", "paper_transformer_mt_lm")
def transformer_mt_lm(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    args.dropout = getattr(args, "dropout", 0.3)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before",
                                            True)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before",
                                            True)

    base_architecture(args)


@register_criterion('cross_entropy_prior')
class PriorLoss(FairseqCriterion):
    """This is a composite loss that, given a list of model outputs and a list of targets,
    computes an average of losses for each output-target pair"""

    def __init__(self, task, sentence_avg, label_smoothing, prior_lambda,
                 prior_tau):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.label_smoothing = label_smoothing
        self.prior_lambda = prior_lambda
        self.prior_tau = prior_tau

        if label_smoothing > 0:
            self.base_criterion = LabelSmoothedCrossEntropyCriterion(task,
                                                                     sentence_avg,
                                                                     label_smoothing)
        else:
            self.base_criterion = CrossEntropyCriterion(task,
                                                        sentence_avg)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--prior-lambda', default=0.1, type=float,
                            help='weight of the prior')
        parser.add_argument('--prior-tau', default=1, type=float,
                            help='temperature of the prior')
        parser.add_argument('--label-smoothing', default=0., type=float,
                            metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def compute_loss(self, model, net_output, prior_output, sample,
                     reduce=True):
        lprobs = F.log_softmax(net_output[0] / self.prior_tau, dim=-1)
        probs = F.softmax(prior_output[0] / self.prior_tau, dim=-1)
        losses = F.kl_div(lprobs, probs, reduction='none').sum(-1)

        mask = ~ model.get_targets(sample, net_output).eq(self.padding_idx)
        losses = losses * mask

        if reduce:
            loss = losses.sum()
        else:
            loss = losses.view(-1)

        # multiply with tau^2 to make loss tau invariant
        loss = loss * (self.prior_tau ** 2)

        return loss, loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # the lm is part of the model so whenever the model is set to train
        # the status of the lm changes as well. Therefore, we have
        # to set it to eval mode in each step
        model.lm.eval()

        net_output = model(**sample['net_input'])
        prior_output = model.lm(sample['net_input']['prev_output_tokens'])

        main_loss, main_nll_loss = self.base_criterion.compute_loss(model,
                                                                    net_output,
                                                                    sample,
                                                                    reduce=reduce)
        prior_loss, prior_nll_loss = self.compute_loss(model,
                                                       net_output, prior_output,
                                                       sample, reduce)

        sample_size = sample['target'].size(0) if self.sentence_avg else sample[
            'ntokens']

        loss = main_loss + self.prior_lambda * prior_loss
        logging_output = {
            'loss': loss.data,
            'nll_loss': main_loss.data,
            'kl_loss': prior_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get('kl_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('kl_loss', kl_loss_sum / sample_size / math.log(2),
                           sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2),
                               ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(
                meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(
                meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
