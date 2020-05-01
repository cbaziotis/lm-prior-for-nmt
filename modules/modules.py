import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F, Dropout2d
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.helpers import length_countdown, \
    coin_flip, id2dist, relax_softmax, lm_fusion
from modules.layers import Attention


class RNNEncoder(nn.Module):
    def __init__(self,
                 embed: nn.Module,
                 rnn_size: int,
                 rnn_layers: int,
                 rnn_dropout: float,
                 inp_dropout: float = 0.0,
                 rnn_bidirectional: bool = False,
                 rnn_layer_norm: bool = False,
                 rnn_type: str = "LSTM",
                 feat_size: int = 0,
                 countdown: bool = False,
                 **kwargs):
        super(RNNEncoder, self).__init__()

        assert rnn_type in ["LSTM", "GRU"]

        self.rnn_layer_norm = rnn_layer_norm
        self.rnn_type = rnn_type
        self.feat_size = feat_size
        self.countdown = countdown

        self.embed = embed
        self.inp_dropout = Dropout2d(inp_dropout)

        input_size = embed.embedding_dim + feat_size

        if self.countdown:
            self.Wt = nn.Parameter(torch.rand(1))
            input_size += 1

        rnn = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn(input_size=input_size,
                       hidden_size=rnn_size,
                       num_layers=rnn_layers,
                       dropout=rnn_dropout if rnn_layers > 1 else 0.,
                       bidirectional=rnn_bidirectional,
                       batch_first=True)

        # define output feature size
        self.hidden_size = rnn_size
        if rnn_bidirectional:
            self.hidden_size *= 2

        if self.rnn_layer_norm:
            self.norm_rnn = nn.LayerNorm(self.rnn.hidden_size, eps=1e-6)

        self.init_weights()

    def init_weights(self):
        # model_init(self)
        pass

    @staticmethod
    def reorder_hidden(hidden, order):
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, order, :], hidden[1][:, order, :]
        else:
            hidden = hidden[:, order, :]
        return hidden

    def representation(self, outputs, hidden, mode="last"):
        """
               Unidirectional           Bidirectional
            [ layer 1 - fwd ]       [ layer 1 - fwd ]
            [ layer 2 - fwd ]       [ layer 1 - bwd ]
                    ...                     ...
                    ...             [ layer N - fwd ]
            [ layer N - fwd ]       [ layer N - bwd ]

        """

        def _last(h):
            # (layer x direction) x batch x dim
            if self.rnn.bidirectional:
                return torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
            else:
                return h[-1, :, :]

        if mode == "last":
            if self.rnn_type == "LSTM":
                x = _last(hidden[0])
            elif self.rnn_type == "GRU":
                x = _last(hidden)
            else:
                raise NotImplementedError
        elif mode == "mean":
            x = outputs.mean(1)
        elif mode == "max":
            x = outputs.max(1)[0]
        else:
            raise NotImplementedError

        return x

    def encode(self, x, lengths, hidden=None):

        batch, max_length, feat_size = x.size()

        # --------------------------------------------
        # sorting
        # --------------------------------------------
        lengths_sorted, sorted_i = lengths.sort(descending=True)
        _, reverse_i = sorted_i.sort()
        x = x[sorted_i]

        if hidden is not None:
            hidden = self.reorder_hidden(hidden, sorted_i)

        # --------------------------------------------
        # forward
        # --------------------------------------------
        if self.countdown:
            ticks = length_countdown(lengths_sorted).float() * self.Wt
            x = torch.cat([x, ticks.unsqueeze(-1)], -1)

        packed = pack_padded_sequence(x, lengths_sorted, batch_first=True)
        out_packed, hidden = self.rnn(packed, hidden)
        out_unpacked, _ = pad_packed_sequence(out_packed,
                                              batch_first=True,
                                              total_length=max_length)
        # --------------------------------------------
        # un-sorting
        # --------------------------------------------
        outputs = out_unpacked[reverse_i]
        hidden = self.reorder_hidden(hidden, reverse_i)

        return outputs, hidden

    def forward(self, x, lengths, hidden=None, c=None):
        # not embedded yet
        if isinstance(x, (torch.LongTensor, torch.cuda.LongTensor)):
            x = self.embed(x)
        x = self.inp_dropout(x)

        if c is not None:
            c = c.unsqueeze(1).expand([x.size(0), x.size(1), c.size(1)])
            x = torch.cat([x, c], dim=2)

        outputs, hidden = self.encode(x, lengths, hidden)

        if self.rnn_layer_norm:
            outputs = self.norm_rnn(outputs)

        return {"outputs": outputs, "hidden": hidden}


class AttSeqDecoder(nn.Module):
    def __init__(self,
                 trg_n_tokens: int,
                 embed: nn.Module,
                 enc_size: int,
                 rnn_size: int,
                 rnn_layers: int,
                 rnn_dropout: float,
                 out_dropout: float,
                 rnn_type: str,
                 outputs_fn: str,
                 tie_projections: bool,
                 attention_fn: str,
                 input_feeding: bool,
                 out_non_linearity: str,
                 out_layer_norm: bool,
                 tau_0: float = 1,
                 learn_tau: bool = False,
                 length_control: bool = False,
                 input_shortcut: bool = False,
                 latent_dim: int = 0,
                 **kwargs):
        super(AttSeqDecoder, self).__init__()

        assert outputs_fn in {"sum", "concat"}

        # ----------------------------------------------------
        # Attributes
        # ----------------------------------------------------
        self.trg_n_tokens = trg_n_tokens
        self.input_feeding = input_feeding
        self.input_shortcut = input_shortcut
        self.learn_tau = learn_tau
        self.length_control = length_control
        self.out_non_linearity = out_non_linearity
        self.out_layer_norm = out_layer_norm
        self.tie_projections = tie_projections
        self.rnn_type = rnn_type
        self.outputs_fn = outputs_fn
        self.gs_beta = 1
        self.latent_dim = latent_dim

        # ----------------------------------------------------
        # Layers
        # ----------------------------------------------------
        self.embed = embed

        # the output size of the ho token: ho = [ h || c]
        self.o_size = self.embed.embedding_dim if tie_projections else rnn_size
        dec_inp_dim = self.embed.embedding_dim
        if self.input_feeding:
            dec_inp_dim += self.o_size

        rnn = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn(input_size=dec_inp_dim,
                       hidden_size=rnn_size,
                       num_layers=rnn_layers,
                       dropout=rnn_dropout if rnn_layers > 1 else 0.,
                       batch_first=True)

        self.out_dropout = nn.Dropout(out_dropout)

        self.attention = Attention(enc_size, rnn_size + latent_dim,
                                   method=attention_fn)

        # learnt temperature parameter
        if self.learn_tau:
            linear_tau = nn.Linear(rnn_size, 1, bias=False)
            self.softplus = nn.Sequential(linear_tau, nn.Softplus())
            self.tau_0 = tau_0

        # -------------------------------------------------------------
        # projection layers
        # -------------------------------------------------------------
        if self.outputs_fn == "sum":
            self.W_c = nn.Linear(enc_size, self.o_size)
            self.W_h = nn.Linear(rnn_size, self.o_size)
            if latent_dim > 0:
                self.W_z = nn.Linear(latent_dim, self.o_size)

            if self.input_shortcut:
                self.W_e = nn.Linear(dec_inp_dim, self.o_size)

        elif self.outputs_fn == "concat":
            _concat_dim = enc_size + rnn_size + latent_dim
            if self.input_shortcut:
                _concat_dim += dec_inp_dim

            self.W_h = nn.Linear(_concat_dim, self.o_size)

        self.logits = nn.Linear(self.o_size, trg_n_tokens)

        if self.out_layer_norm:
            self.norm_outs = nn.LayerNorm(self.o_size, eps=1e-6)

        self.init_weights()
        self.tie_weights()

    def init_weights(self):
        pass

    def tie_weights(self):
        if self.tie_projections:
            self.logits.weight = self.embed.embedding.weight
            # self.embed.embedding.weight = self.logits.weight

    def _step_emb(self, step, trg, logits, sampling_prob, sampling_mode, tau):
        """
        Get the token embedding for the current timestep. Possible options:
        - select the embedding by a given index
        - sample a token from a probability distribution and embed
        - construct a "fuzzy" embedding, by taking a convex combination of all
        the token embeddings, parameterized by a probability distribution

        Note: At the last timestep (step == max_length), when by definition
        there is not a target word that is given to us, generation a distribution
        from the logits regardless of whether the model is trained with
        teacher-forcing, scheduled-sampling and/or gumbel-softmax.

        """
        batch, max_length = trg.size()

        if sampling_prob == 1 or coin_flip(sampling_prob) or step == max_length:
            assert sampling_mode in ["argmax", "gs", "st", "gs-st", "softmax"]

            # get the argmax
            if sampling_mode == "argmax":
                maxv, maxi = logits.max(dim=2)
                e_i = self.embed(maxi)
                return e_i, id2dist(maxi, self.logits.out_features).unsqueeze(1)

            # get the expected embedding, parameterized by the posterior
            elif sampling_mode in ["gs", "st", "gs-st", "softmax"]:

                # add gumbel noise only during training
                _add_gumbel = self.training and sampling_mode in ["gs", "gs-st"]

                # discretize the distributions if Straight-Trough is used
                hard = sampling_mode in ["st", "gs-st"]

                # make sure not to generate <pad>,
                # because it's a zero embedding and we'll get no gradients.
                pad_mask = torch.zeros_like(logits)
                pad_mask[:, :, 0] = torch.min(logits)
                logits = logits + pad_mask

                dist = relax_softmax(logits, tau, gumbel=_add_gumbel, hard=hard,
                                     beta=self.gs_beta)
                e_i = self.embed.expectation(dist)
                return e_i, dist
            else:
                raise NotImplementedError
        else:
            w_i = trg[:, step].unsqueeze(1)
            e_i = self.embed(w_i)
            return e_i, id2dist(w_i, self.logits.out_features).unsqueeze(1)

    def _step_input(self, embeddings, input_feed=None, tick=None):
        """
        Create the input to the decoder for a given step
        """
        batch = embeddings.size(0)
        _input = embeddings

        if self.input_feeding:
            if input_feed is None:
                with torch.no_grad():
                    input_feed = torch.zeros(batch, 1, self.o_size,
                                             dtype=_input.dtype,
                                             device=_input.device)
            _input = torch.cat((embeddings, input_feed), -1)
        if self.length_control:
            _input = torch.cat((_input, tick), -1)

        return _input

    def _step(self, inp, enc_outputs, state, src_mask, latent=None):
        """
        Perform one decoding step.
        1. Feed the input to the decoder and obtain the contextualized
            token representations.
        2. Generate a context vector. It is a convex combination of the
            states of the encoder, the weights of which are a function of each
            state of the encoder and the current state of the decoder.
        3. Re-weight the decoder's state with the context vector.
        4. Project the context-aware vector to the vocabulary.

        """
        # 1. Feed the input to the decoder
        outputs, state = self.rnn(inp, state)

        # 2. Generate the context vector
        query = outputs.squeeze(1)
        if latent is not None:
            query = torch.cat([query, latent], 1)
        contexts, att_scores = self.attention(enc_outputs, query, src_mask)

        # apply dropout before combining the features
        outputs = self.out_dropout(outputs)
        contexts = self.out_dropout(contexts)

        # 3. Re-weight the decoder's state with the context vector.
        if self.outputs_fn == "sum":
            o = self.W_h(outputs) + self.W_c(contexts)

            if self.input_shortcut:
                o = o + self.W_e(inp)

            if self.latent_dim > 0:
                o = o + self.W_z(latent)

        elif self.outputs_fn == "concat":
            o = torch.cat([outputs, contexts], 2)

            if self.input_shortcut:
                o = torch.cat([o, inp], 2)

            if self.latent_dim > 0:
                o = torch.cat([o, latent.unsqueeze(1)], 2)

            o = self.W_h(o)


        else:
            raise ValueError

        if self.out_layer_norm:
            o = self.norm_outs(o)

        if self.out_non_linearity == "relu":
            o = torch.relu(o)
        elif self.out_non_linearity == "tanh":
            o = torch.tanh(o)

        # 4. Project the context-aware vector to the vocabulary.
        logits = self.logits(o)

        return logits, outputs, state, o, att_scores

    def forward(self,
                trg: Tensor,
                enc_outputs: Tensor,
                init_hidden,
                enc_lengths: Tensor,
                src_mask: Tensor,
                trg_lengths: Tensor = None,
                word_dropout=0,
                sampling=0.0,
                sampling_mode="argmax",
                tau=1.0,
                lm: nn.Module = None,
                fusion=None,
                fusion_a=None,
                lm_state=None,
                latent=None,
                **kwargs):
        """

        Returns:
            Note: dists contain one less element than logits, because
            we do not care about sampling from the last timestep as it will not
            be used for sampling another token.
            The last timestep should correspond to the EOS token, and the
            corresponding logit will be used only for computing the NLL loss
            of the EOS token.

        When the decoder is used for supervised learning you can use
        the given x_i tokens (teacher-forcing). However, you can also sample
        the next token (scheduled-sampling), or even approximate it with
        the gumbel-relaxation and back-propagate through the sample.

        For unsupervised learning you may need the distributions p_i that
        generated the samples.

        dists         p_0 ~─┐   p_1  ~── ...      p_n
        (sampling)     ~    │    ~       ...       ~
        logits        u_0   │   u_1      ...      u_n
                       ↑    │    ↑       ...       ↑
                      c_0   │   c_1      ...      c_n
                       ↑    │    ↑       ...       ↑
        outputs       h_0   │   h_1      ...      h_n
                       ↑    │    ↑       ...       ↑
                      e_0   └─> e_1      ... ──>  e_n

                       ↑         ↑                 ↑
                      x_0       x_1               x_n
                     (<s>)

        """
        results = {"logits": [], "outputs": [], "attention": [],
                   "dists": [], "gate": [], "tau": [], "samples": [],
                   "dec": [], "lm": []}
        batch, max_length = trg.size()

        # ------------------------------------------------------------------
        # Prepare Decoding
        # ------------------------------------------------------------------
        # initial hidden state of the decoder, and initial context
        state_i = init_hidden
        context_i, o_i, tick = None, None, None

        # if we are doing inference, then simply take the argmax
        if self.training is False:
            sampling_mode = "argmax"

        # pre-compute source state projections for efficiency
        if self.attention.method in ["general", "additive"]:
            self.attention.precompute_enc_projections(enc_outputs)

        if self.length_control:
            countdown = length_countdown(trg_lengths).float() * self.W_tick
            ratio = trg_lengths.float() / enc_lengths.float()

        # At the first step (step==0) select the embedding of the given
        # target word (usually the <sos> token).
        e_i, d_i = self._step_emb(step=0, trg=trg, logits=None, sampling_prob=0,
                                  sampling_mode="argmax", tau=1)

        # ------------------------------------------------------------------
        # Decoding Loop
        # ------------------------------------------------------------------
        for i in range(max_length):

            # ---------------------------------------------------------
            # 1. construct time-step input
            # ---------------------------------------------------------
            if i > 0:
                e_i = F.dropout2d(e_i, word_dropout, self.training)

            # the number of remaining tokens
            if self.length_control:
                tick = torch.stack((countdown[:, i], ratio), -1).unsqueeze(1)

            input_i = self._step_input(e_i, o_i, tick)

            # ---------------------------------------------------------
            # 2. perform one decoding step
            # ---------------------------------------------------------
            step_i = self._step(input_i, enc_outputs, state_i, src_mask, latent)
            logits_i, outs_i, state_i, o_i, att_i = step_i

            # ---------------------------------------------------------
            # 3. obtain the NEXT input word embedding
            # ---------------------------------------------------------

            # feed the input to the prior and interpolate with the decoder
            if fusion is not None:
                assert lm is not None
                _len = torch.ones(batch, device=e_i.device, dtype=torch.long)
                lm_outs = lm(d_i.max(dim=2)[1], _len, lm_state)
                lm_logits, lm_state = lm_outs["logits"], lm_outs["hidden"]
                results["lm"].append(lm_logits)
                results["dec"].append(logits_i)
                logits_i = lm_fusion(logits_i, lm_logits, fusion, fusion_a)

            # generation the temperature value for the next sampling step
            if self.learn_tau and self.training:
                tau = 1 / (self.softplus(outs_i.squeeze()) + self.tau_0)
                results["tau"].append(tau)

            # select or sample the next input
            e_i, d_i = self._step_emb(step=i + 1, trg=trg, logits=logits_i,
                                      sampling_prob=sampling,
                                      sampling_mode=sampling_mode, tau=tau)

            # ---------------------------------------------------------
            results["logits"].append(logits_i)
            results["outputs"].append(outs_i)
            results["attention"].append(att_i.unsqueeze(1))
            results["dists"].append(d_i)
            results["samples"].append(e_i)

        results = {k: torch.cat(v, dim=1) if len(v) > 0 else None
                   for k, v in results.items()}

        return results
