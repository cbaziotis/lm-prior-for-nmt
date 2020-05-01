import torch
from torch import nn

from libs.joeynmt.decoders import TransformerDecoder
from libs.joeynmt.encoders import TransformerEncoder
from libs.joeynmt.helpers import subsequent_mask
from modules.beam import beam_search
from modules.helpers import fake_inputs, sequence_mask, \
    lm_fusion, infer_sample_lengths
from modules.layers import RNNBridge, Embed
from modules.modules import AttSeqDecoder, RNNEncoder


class RNNLM(nn.Module):
    def __init__(self, ntokens,
                 tie_projections=False,
                 out_layer_norm=False,
                 **kwargs):
        super(RNNLM, self).__init__()

        self.tie_projections = tie_projections
        self.out_layer_norm = out_layer_norm

        self.embed = Embed(ntokens,
                           kwargs["emb_size"],
                           emb_dropout=kwargs["emb_dropout"],
                           trainable=kwargs["emb_trainable"],
                           layer_norm=kwargs["emb_layer_norm"],
                           max_norm=kwargs["emb_max_norm"])

        self.encoder = RNNEncoder(self.embed, **kwargs)
        assert not self.encoder.rnn.bidirectional

        enc_dim = self.encoder.hidden_size
        emb_dim = self.embed.embedding_dim

        self.project_down = (self.tie_projections and enc_dim != emb_dim)

        if self.project_down:
            self.W_h = nn.Linear(enc_dim, emb_dim)
            out_dim = emb_dim
        else:
            out_dim = enc_dim

        if self.out_layer_norm:
            self.LN_h = nn.LayerNorm(out_dim, eps=1e-6)

        self.logits = nn.Linear(out_dim, ntokens)

        self.init_weights()
        self.tie_weights()

    def init_weights(self):
        pass

    def tie_weights(self):
        if self.tie_projections:
            self.logits.weight = self.embed.embedding.weight

    def forward(self, x, lengths, hidden=None, c=None):
        results = self.encoder(x, lengths, hidden, c)

        if self.project_down:
            results["outputs"] = self.W_h(results["outputs"])

        if self.out_layer_norm:
            results["outputs"] = self.LN_h(results["outputs"])

        results["logits"] = self.logits(results["outputs"])

        return results


class TransformerLM(nn.Module):

    def __init__(self, ntoken,
                 emb_size=512,
                 nhead=8,
                 nhid=2048,
                 nlayers=6,
                 dropout=0.1,
                 tie_projections=True,
                 **kwargs):
        super(TransformerLM, self).__init__()

        self.tie_projections = tie_projections
        self.ninp = emb_size

        self.embed = Embed(ntoken, emb_size, scale=True)
        self.encoder = TransformerEncoder(hidden_size=emb_size,
                                          ff_size=nhid,
                                          num_layers=nlayers,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          emb_dropout=dropout)
        self.logits = nn.Linear(emb_size, ntoken)

        self.tie_weights()

    def tie_weights(self):
        if self.tie_projections:
            self.logits.weight = self.embed.embedding.weight

    def forward(self, x, lengths):
        emb = self.embed(x)

        # mask padded + future steps
        pad_mask = sequence_mask(lengths, x.size(1)).unsqueeze(1)
        mask = pad_mask & subsequent_mask(x.size(1)).type_as(pad_mask)

        states = self.encoder(emb, None, mask)[0]
        logits = self.logits(states)

        return {"logits": logits}


class Seq2SeqBase(nn.Module):

    def __init__(self, src_n_tokens, trg_n_tokens, **kwargs):
        super(Seq2SeqBase, self).__init__()

        self.src_n_tokens = src_n_tokens
        self.trg_n_tokens = trg_n_tokens

    def init_weights(self):
        pass

    def tie_weights(self):
        pass

    def encode(self, src_inp, src_lengths):
        raise NotImplementedError

    def decode(self, trg_inp, enc_outs, dec_init, src_lengths, src_mask,
               trg_lengths, **kwargs):
        raise NotImplementedError

    def translate(self, src, src_lengths, sos_id, trg_lengths=None, **kwargs):
        raise NotImplementedError

    def beam(self, src, src_lengths, sos_id, eos_id, pad_id,
             beam_size, length_penalty, **kwargs):
        raise NotImplementedError

    def forward(self, src, trg, src_lengths, trg_lengths, **kwargs):
        raise NotImplementedError


class Seq2SeqRNN(Seq2SeqBase):

    def __init__(self, src_n_tokens, trg_n_tokens, **kwargs):
        super(Seq2SeqRNN, self).__init__(src_n_tokens, trg_n_tokens)

        self.bridge_non_linearity = kwargs["bridge_non_linearity"]
        self.ninp = kwargs["encoder"]["emb_size"]  # for noam optimizer

        self.embed_src = Embed(self.src_n_tokens,
                               kwargs["encoder"]["emb_size"],
                               emb_dropout=kwargs["encoder"]["emb_dropout"],
                               trainable=kwargs["encoder"]["emb_trainable"],
                               layer_norm=kwargs["encoder"]["emb_layer_norm"],
                               padding_idx=kwargs.get("enc_padding_idx", 0))

        self.embed_tgt = Embed(self.trg_n_tokens,
                               kwargs["decoder"]["emb_size"],
                               emb_dropout=kwargs["decoder"]["emb_dropout"],
                               trainable=kwargs["decoder"]["emb_trainable"],
                               layer_norm=kwargs["decoder"]["emb_layer_norm"],
                               padding_idx=kwargs.get("dec_padding_idx", 0))

        self.encoder = RNNEncoder(self.embed_src, **kwargs["encoder"])

        self.decoder = AttSeqDecoder(self.trg_n_tokens,
                                     self.embed_tgt,
                                     self.encoder.hidden_size,
                                     **kwargs["decoder"])

        self.bridge = RNNBridge(enc_dim=self.encoder.hidden_size,
                                dec_dim=self.decoder.rnn.hidden_size,
                                dec_layers=self.decoder.rnn.num_layers,
                                dec_type=self.decoder.rnn.mode)

    def tie_weights(self):
        self.decoder.tie_weights()

    def encode(self, src_inp, src_lengths):
        return self.encoder(src_inp, src_lengths)

    def decode(self, trg_inp, enc_outs, dec_init, src_lengths, src_mask,
               trg_lengths, **kwargs):
        # logits, outputs, hidden, (attentions, ctx_outputs, words, embs)
        outputs = self.decoder(trg_inp, enc_outs, dec_init,
                               src_lengths, src_mask, **kwargs)

        return outputs

    def init_decoder(self, enc_outputs, enc_hidden):
        enc_state = self.encoder.representation(enc_outputs, enc_hidden)
        if self.bridge_non_linearity == "tanh":
            enc_state = torch.tanh(enc_state)
        elif self.bridge_non_linearity == "relu":
            enc_state = torch.relu(enc_state)
        else:
            raise NotImplementedError
        return self.bridge(enc_state)

    def translate(self, x, x_lengths, sos_id, y_lengths=None, **kwargs):

        enc = self.encode(x, x_lengths)
        dec_init = self.init_decoder(enc["outputs"], enc["hidden"])

        # Set the target length larger than source.
        # It will be pruned after the EOS anyway.
        if y_lengths is None:
            y_lengths = (x_lengths.float() * 1.5).long()

        src_mask = sequence_mask(x_lengths, x.size(1))
        inp_fake = fake_inputs(x, y_lengths, sos_id)
        dec = self.decode(inp_fake, enc["outputs"], dec_init,
                          x_lengths, src_mask, y_lengths,
                          sampling=1, sampling_mode="argmax", **kwargs)

        return enc, dec

    def beam(self, x, x_len, sos_id, eos_id, pad_id,
             beam_size, length_penalty, **kwargs):

        enc = self.encode(x, x_len)
        dec_init = self.init_decoder(enc["outputs"], enc["hidden"])
        src_mask = sequence_mask(x_len, x.size(1))

        outputs = beam_search(
            decoder=self.decoder,
            size=beam_size,
            bos_index=sos_id,
            eos_index=eos_id,
            pad_index=pad_id,
            encoder_output=enc["outputs"],
            encoder_hidden=dec_init,
            src_mask=src_mask,
            max_output_length=(x_len.float() * 1.5).long().max(),
            alpha=length_penalty,
            lm_hidden=None,
            **kwargs
        )

        return outputs

    def forward(self, src, trg, src_lengths, trg_lengths, **kwargs):
        enc = self.encode(src, src_lengths)
        src_mask = sequence_mask(src_lengths, src.size(1))
        dec_init = self.init_decoder(enc["outputs"], enc["hidden"])
        dec = self.decode(trg, enc["outputs"], dec_init,
                          src_lengths, src_mask, trg_lengths, **kwargs)

        return enc, dec


class Seq2SeqTransformer(Seq2SeqBase):

    def __init__(self,
                 src_n_tokens,
                 trg_n_tokens,
                 emb_size=512,
                 nhead=8,
                 nhid=2048,
                 nlayers=6,
                 dropout=0.1,
                 tie_projections=True,
                 **kwargs
                 ):
        super(Seq2SeqTransformer, self).__init__(src_n_tokens, trg_n_tokens,
                                                 **kwargs)

        self.ninp = emb_size

        self.tgt_mask = None
        self.tie_projections = tie_projections

        self.embed_src = Embed(self.src_n_tokens, emb_size, scale=True,
                               padding_idx=kwargs.get("enc_padding_idx", 0))
        self.embed_tgt = Embed(self.trg_n_tokens, emb_size, scale=True,
                               padding_idx=kwargs.get("dec_padding_idx", 0))

        self.encoder = TransformerEncoder(hidden_size=emb_size,
                                          ff_size=nhid,
                                          num_layers=nlayers,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          emb_dropout=dropout)

        self.decoder = TransformerDecoder(num_layers=nlayers,
                                          num_heads=nhead,
                                          hidden_size=emb_size,
                                          ff_size=nhid,
                                          dropout=dropout,
                                          emb_dropout=dropout,
                                          vocab_size=trg_n_tokens)

        self.tie_weights()

    def tie_weights(self):
        if self.tie_projections:
            self.decoder.output_layer.weight = self.embed_tgt.embedding.weight

    def encode(self, x, lengths, **kwargs):
        emb = self.embed_src(x)
        pad_mask = sequence_mask(lengths, x.size(1)).unsqueeze(1)
        memory = self.encoder(emb, None, pad_mask)[0]
        return memory, pad_mask

    def decode(self, y, memory, src_mask, y_lengths, **kwargs):
        y_emb = self.embed_tgt(y)

        if y_lengths is None:
            trg_mask = src_mask.new_ones([1, 1, 1])
        else:
            trg_mask = sequence_mask(y_lengths, y.size(1)).unsqueeze(1)

        output, states = self.decoder(trg_embed=y_emb,
                                      encoder_output=memory,
                                      src_mask=src_mask,
                                      trg_mask=trg_mask)[:2]

        return output, states

    def translate(self, x, x_lengths, sos_id, y_lengths=None,
                  lm: nn.Module = None, fusion=None, fusion_a: float = None,
                  **kwargs):

        memory, src_mask = self.encode(x, x_lengths)

        if y_lengths is None:
            y_len = round(x.size(1) * 1.5)
        else:
            y_len = max(y_lengths)

        y = torch.zeros_like(x[:, :1]).fill_(sos_id)

        for i in range(y_len):
            y_lengths = infer_sample_lengths(y, eos_id=2)
            logits, output = self.decode(y, memory, src_mask, y_lengths)

            if lm is not None:
                lm_logits = lm(y, y_lengths)["logits"]
                logits = lm_fusion(logits, lm_logits, fusion, fusion_a)

            preds = logits[:, -1:, :].max(dim=2)[1]
            y = torch.cat([y, preds], dim=-1)

        return {"memory": memory}, {"logits": logits}

    def beam(self, src, src_lengths, sos_id, eos_id, pad_id,
             beam_size, length_penalty, **kwargs):

        memory, src_key_padding_mask = self.encode(src, src_lengths)

        outputs = beam_search(
            decoder=self.decoder,
            size=beam_size,
            bos_index=sos_id,
            eos_index=eos_id,
            pad_index=pad_id,
            encoder_output=memory,
            encoder_hidden=memory,
            src_mask=src_key_padding_mask,
            max_output_length=(src_lengths.float() * 1.3).long().max(),
            alpha=length_penalty,
            lm_hidden=None,
            embed_tgt=self.embed_tgt,
            **kwargs
        )

        return outputs

    def forward(self, x, y, x_len, y_len,
                lm: nn.Module = None, fusion=None, fusion_a: float = None,
                **kwargs):
        enc = dict()
        dec = dict()

        memory, src_mask = self.encode(x, x_len)
        enc["memory"] = memory

        logits, states = self.decode(y, memory, src_mask, y_len)

        if lm is not None:
            lm_logits = lm(y, y_len)["logits"]

            dec["dec"] = logits
            dec["lm"] = lm_logits

            logits = lm_fusion(logits, lm_logits, fusion, fusion_a)

        dec["logits"] = logits

        return enc, dec
