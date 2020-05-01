import hashlib
import json
import os
from collections.__init__ import Counter

import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from gensim.models import FastText
from sacremoses import MosesDetokenizer
from sklearn import preprocessing
from numpy import linalg as LA
from tqdm import tqdm
from transformers import GPT2Tokenizer

from helpers.emb_utils import load_embeddings


class Vocab(object):
    """
    The Vocab Class, holds the vocabulary of a corpus and
    mappings from tokens to indices and vice versa.
    """

    def __init__(self, pad="<pad>", sos="<sos>", eos="<eos>", unk="<unk>",
                 oovs=0, file=None, preprocess=None, subword=None, lang=None):
        self.PAD = pad
        self.SOS = sos
        self.EOS = eos
        self.UNK = unk

        self.PAD_id = 0
        self.SOS_id = None
        self.EOS_id = None
        self.UNK_id = None

        self.lang = lang

        self.oovs = oovs

        self.vocab = Counter()

        self.tok2id = dict()
        self.id2tok = dict()
        self.freqs = dict()

        self.gpt2_tokenizer = None
        self.is_gpt2 = False

        self.subword = subword

        if file is not None:
            assert preprocess is not None, "Need preprocess() to build vocab!"
            self.build(file, preprocess)

    def reset(self):
        self.tok2id = dict()
        self.id2tok = dict()

        self._add_special_tokens()
        self._set_special_token_ids()
        self.vocab = Counter()

    def from_vocab_instance(self, vocab):
        for attr, value in vars(vocab).items():
            setattr(self, attr, value)
        self._set_special_token_ids()
        return self

    def from_gpt2(self, tokenizer: GPT2Tokenizer):
        self.SOS = tokenizer.bos_token
        self.EOS = tokenizer.eos_token
        self.UNK = tokenizer.unk_token
        self.SOS_id = tokenizer.bos_token_id
        self.EOS_id = tokenizer.eos_token_id
        self.UNK_id = tokenizer.unk_token_id

        self.tok2id = tokenizer.encoder
        self.id2tok = tokenizer.decoder
        self.gpt2_tokenizer = tokenizer

        self.is_gpt2 = True

    def gpt2_tok(self, x):
        tokens = self.gpt2_tokenizer.convert_tokens_to_ids(
            self.gpt2_tokenizer.tokenize(x, add_prefix_space=True))
        return [self.SOS_id] + tokens + [self.EOS_id]

    def gpt2_detok(self, x):
        return self.gpt2_tokenizer.convert_tokens_to_string(x)

    def detokenize(self, x):
        if self.is_gpt2:
            return self.gpt2_detok(x).lstrip()
        if self.subword is not None:
            return ''.join(x).replace('▁', ' ').lstrip().rstrip()
        elif self.lang is not None:
            return MosesDetokenizer(lang=self.lang).detokenize(x)
        else:
            return ' '.join(x)

    def read_sequence(self, tokens):
        self.vocab.update(tokens)

    def read_embeddings(self, embeddings, word2id):
        """
        Create an Embeddings Matrix, in which each row corresponds to
        the word vector from the pretrained word embeddings.
        If a word is missing from the provided pretrained word vectors, then
        sample a new embedding, from the gaussian of the pretrained embeddings.

        """

        mu = embeddings.mean(axis=0)
        sigma = embeddings.std(axis=0)

        filtered_embeddings = numpy.zeros((len(self), embeddings.shape[1]))

        mask = numpy.zeros(len(self))
        missing = []

        for token_id, token in tqdm(self.id2tok.items(),
                                    desc="Reading embeddings...",
                                    total=len(self.id2tok.items())):
            if token not in word2id or token == "<unk>":
                sample = numpy.random.normal(mu, sigma / 4)
                filtered_embeddings[token_id] = sample

                mask[token_id] = 1
                missing.append(token_id)
            else:
                filtered_embeddings[token_id] = embeddings[word2id[token]]

        print()
        print(f"Missing tokens from the pretrained embeddings: {len(missing)}")

        return filtered_embeddings, mask, missing

    def read_fasttext(self, file):
        """
        Create an Embeddings Matrix, in which each row corresponds to
        the word vector from the pretrained word embeddings.
        If a word is missing then obtain a representation on-the-fly
        using fasttext.

        Args:
            file:
            dim:

        Returns:

        """
        model = FastText.load_fasttext_format(file)

        embeddings = numpy.zeros((len(self), model.vector_size))

        missing = []

        for token_id, token in tqdm(self.id2tok.items(),
                                    desc="Reading embeddings...",
                                    total=len(self.id2tok.items())):
            if token not in model.wv.vocab:
                missing.append(token)
            embeddings[token_id] = model[token]

        print(f"Missing tokens from the pretrained embeddings: {len(missing)}")

        return embeddings, missing

    def add_token(self, token):
        index = len(self.tok2id)

        if token not in self.tok2id:
            self.tok2id[token] = index
            self.id2tok[index] = token

    def __hash__(self):
        return hashlib.sha256(
            json.dumps(self.tok2id, sort_keys=True).encode()).hexdigest()

    def hash(self):
        return hashlib.sha256(
            json.dumps(self.tok2id, sort_keys=True).encode()).hexdigest()

    def _set_special_token_ids(self):
        self.PAD_id = self.tok2id.get(self.PAD, 0)
        self.SOS_id = self.tok2id[self.SOS]
        self.EOS_id = self.tok2id[self.EOS]
        self.UNK_id = self.tok2id[self.UNK]

    def _add_special_tokens(self):
        self.add_token(self.PAD)
        self.add_token(self.SOS)
        self.add_token(self.EOS)
        self.add_token(self.UNK)

        for i in range(self.oovs):
            self.add_token(f"<oov-{i}>")

    def build(self, file, preprocess, suffix='.vocab'):
        """
        Build the vocab from a txt corpus.
        The function assumes that the txt file contains one sentence per line.
        Afterwards, write the vocab data to disk as {file}{suffix}.
        """
        vocab_file = file + suffix
        if os.path.exists(vocab_file):
            self.load_from_vocab_file(vocab_file)
        else:
            self._add_special_tokens()
            self._set_special_token_ids()
            with open(file, encoding="utf-8") as f:
                for line in f:
                    self.read_sequence(preprocess(line))
            self.save(vocab_file)

    def build_from_tokens(self, dataset):
        """
        Build the vocab from a list of wordlists.
        """
        self._add_special_tokens()
        self._set_special_token_ids()
        for tokens in dataset:
            self.read_sequence(tokens)
        self.build_lookup()

    def load_from_vocab_file(self, file):
        """
        Load vocabulary from a .vocab file
        """

        self.tok2id = dict()
        self.id2tok = dict()

        self._add_special_tokens()
        self._set_special_token_ids()
        self.vocab = Counter()

        for line in open(file, encoding="utf-8").readlines():
            token, count = line.split("\t")
            self.vocab[token] = float(count)
            self.add_token(token)

    def save(self, file):
        with open(file, "w", encoding="utf-8") as f:
            for w, k in self.vocab.most_common():
                f.write("\t".join([w, str(k)]) + "\n")

    def is_corrupt(self):
        return len([tok for tok, index in self.tok2id.items()
                    if self.id2tok[index] != tok]) > 0

    def get_tokens(self):
        return [self.id2tok[key] for key in sorted(self.id2tok.keys())]

    def build_lookup(self, size=None):
        self.tok2id = dict()
        self.id2tok = dict()

        self._add_special_tokens()
        self._set_special_token_ids()

        for w, k in self.vocab.most_common(size):
            self.add_token(w)

    def visualize_vocab(self):
        sns.distplot(list(self.vocab.values()), bins=50, kde=False)
        plt.show()

    def get_freqs(self):
        _sum = sum(self.vocab.values())
        freqs = dict()
        for i in range(len(self)):
            tok = self.id2tok[i]
            freqs[tok] = self.vocab[tok] / _sum

        return freqs

    def get_freqs_list(self):
        freqs = self.get_freqs()
        return [freqs[self.id2tok[i]] for i in range(len(self))]

    def sos_id(self):
        return self.tok2id[self.SOS]

    def __len__(self):
        return len(self.tok2id)
