import os

from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from helpers.opts import exp_options
from helpers.training import load_checkpoint
from helpers.transfer import freeze_module
from models.nmt_prior_callbacks import EvalCallback, SamplesCallback, \
    AttentionCallback
from models.nmt_prior_helpers import nmt_dataloaders, \
    prior_model_from_checkpoint, eval_best, backtranslate
from models.nmt_prior_trainer import NmtPriorTrainer
from modules.callbacks import LossCallback, GradientCallback, \
    ModuleGradientCallback, FunctionCallback
from modules.data.vocab import Vocab
from modules.initializations import model_init
from modules.models import Seq2SeqTransformer, Seq2SeqRNN
from sys_config import MODEL_CNF_DIR


def run(config):
    # -------------------------------------------------------------------------
    # Load pretrained models
    # -------------------------------------------------------------------------
    vocab_src = None
    vocab_trg = None

    # Load pretrained LM, which will be used for LM-Fusion or as LM-prior
    if config["data"]["prior_path"] is not None:
        if "gpt2" in config["data"]["prior_path"]:
            _gpt_model = os.path.split(config["data"]["prior_path"])[1]
            tokenizer = GPT2Tokenizer.from_pretrained(_gpt_model)
            vocab_trg = Vocab()
            vocab_trg.from_gpt2(tokenizer)
            _checkp_prior = GPT2LMHeadModel.from_pretrained(_gpt_model)
            config["model"]["dec_padding_idx"] = None
        else:
            _checkp_prior = load_checkpoint(config["data"]["prior_path"])
            vocab_trg = _checkp_prior["vocab"]

            if _checkp_prior["config"]["data"]["subword_path"] is not None:
                sub_path = _checkp_prior["config"]["data"]["subword_path"]
                config["data"]["trg"]["subword_path"] = sub_path

    # -------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # -------------------------------------------------------------------------
    train_loader, val_loader = nmt_dataloaders(config, vocab_src, vocab_trg)

    # -------------------------------------------------------------------------
    # Initialize Model and Priors
    # -------------------------------------------------------------------------
    model_type = config["model"].get("type", "rnn")
    src_ntokens = len(val_loader.dataset.src.vocab)
    trg_ntokens = len(val_loader.dataset.trg.vocab)

    # Initialize Model
    if model_type == "rnn":
        model = Seq2SeqRNN(src_ntokens, trg_ntokens, **config["model"])
    elif model_type == "transformer":
        model = Seq2SeqTransformer(src_ntokens, trg_ntokens, **config["model"])
    else:
        raise NotImplementedError

    model_init(model, **config.get("init", {}))

    # Initialize prior LM
    _has_lm_prior = "prior" in config["losses"]
    _has_lm_fusion = config["model"]["decoding"].get("fusion") is not None
    if _has_lm_prior or _has_lm_fusion:
        if "gpt2" in config["data"]["prior_path"]:
            prior = _checkp_prior
            prior.to(config["device"])
            freeze_module(prior)
            for name, module in prior.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0
        else:
            prior = prior_model_from_checkpoint(_checkp_prior)
            prior.to(config["device"])
            freeze_module(prior)
    else:
        prior = None

    model.tie_weights()

    # -------------------------------------------------------------------------
    # Training Pipeline
    # -------------------------------------------------------------------------
    callbacks = [
        LossCallback(config["logging"]["log_interval"]),
        GradientCallback(config["logging"]["log_interval"]),
        ModuleGradientCallback(["encoder"], config["logging"]["log_interval"]),
        SamplesCallback(config["logging"]["log_interval"]),
        EvalCallback(config["logging"]["eval_interval"], keep_best=True,
                     early_stop=config["optim"]["early_stop"])
    ]
    if model_type == "rnn":
        callbacks.append(AttentionCallback(config["logging"]["eval_interval"]))

    eval_interval = config["logging"]["eval_interval"]
    full_eval_interval = config["logging"].get("full_eval_interval",
                                               15 * eval_interval)
    callbacks.append(FunctionCallback(eval_best, full_eval_interval))

    trainer = NmtPriorTrainer(model, train_loader, val_loader, config,
                              config["device"],
                              prior=prior, callbacks=callbacks,
                              src_dirs=config["src_dirs"],
                              resume_state_id=config["resume_state_id"])

    if trainer.exp.has_finished():
        return trainer

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch()
        val_loss = trainer.eval_epoch()
        print("\n" * 3)

        if trainer.early_stop:
            print("Stopping early ...")
            break

    trainer.exp.finalize()
    return trainer


if __name__ == "__main__":
    _config = exp_options(os.path.join(MODEL_CNF_DIR,
                                       # "seq2seq_proto_elen.yaml"
                                       "seq2seq_proto_elen_trans.yaml"
                                       ))
    # _config["cores"] = 0
    # _config["pin_memory"] = False
    trained_model = run(_config)

    eval_best(trained_model)

    if "backtranslate_path" in trained_model.config["data"]:
        backtranslate(trained_model)
