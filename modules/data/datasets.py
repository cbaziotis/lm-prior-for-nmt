import hashlib
import inspect
import os
import pickle
from abc import ABC

import numpy
import sentencepiece as spm
from sacremoses import MosesTokenizer
from tabulate import tabulate
from torch.utils.data import Dataset

from helpers.generic import number_h
from modules.data.streaming import DatasetCache
from modules.data.utils import vectorize, fix_paths
from modules.data.vocab import Vocab


class BaseSequenceDataset(Dataset, ABC):
    def __init__(self,
                 input,
                 tokenize=None,
                 vocab=None,
                 vocab_size=None,
                 subword_path=None,
                 seq_len=0,
                 sos=False,
                 oovs=0,
                 lang="en",
                 subsample=0,
                 **kwargs):
        """
        Base Dataset for Language Modeling.

        Args:
            tokenize (callable): tokenization callable, which takes as input
                a string and returns a list of tokens
            input (str, list): the path to the data file, or a list of samples.
            vocab (Vocab): a vocab instance. If None, then build a new one
                from the Datasets data.
            vocab_size(int): if given, then trim the vocab to the given number.
        """
        self.input = input
        self.seq_len = seq_len
        self.subword_path = subword_path
        self.sos = sos
        self.oovs = oovs
        self.subsample = subsample

        # > define tokenization to be used -------------------------------
        if tokenize is not None:
            self.tokenize = tokenize
        else:
            self.tokenize = self.space_tok

        if self.subword_path is not None:
            subword = spm.SentencePieceProcessor()
            subword_path = fix_paths(subword_path)
            subword.Load(subword_path + ".model")
            self.tokenize = lambda x: subword.EncodeAsPieces(x.rstrip())
        else:
            self.tokenize = MosesTokenizer(lang=lang).tokenize

        # > Build Vocabulary --------------------------------------------
        self.vocab, is_vocab_built = self.init_vocab(vocab, subword_path, oovs)

        # > Cache text file ---------------------------------------------
        self.lengths = []
        _is_cached = False

        def _line_callback(x):
            _tokens = self.tokenize(x)
            self.lengths.append(len(self.add_special_tokens(_tokens)))

            if is_vocab_built is False:
                self.vocab.read_sequence(_tokens)

        # -------------------------------------------------------------
        # If there is a (vocab, lengths) tuple associated with the given input
        # file, then load them from cache and skip the recalculation
        # -------------------------------------------------------------
        _ckey = self._get_cache_key(input, vocab, self.tokenize,
                                    subword_path, vocab_size, self.subsample)
        _cfile = os.path.join(os.path.dirname(input), f".cache_{_ckey}")
        if os.path.isfile(_cfile):
            print("Loading data from cache...", end=" ")
            with open(_cfile, "rb") as f:
                _vocab, self.lengths = pickle.load(f)
                self.vocab = Vocab().from_vocab_instance(_vocab)
            print("done!")
            _is_cached = True

        # > Preprocessing ---------------------------------------------
        print("Preprocessing...")
        self.data = DatasetCache(input,
                                 callback=_line_callback,
                                 subsample=subsample)

        # if the text file has already been cached,
        # but lengths and vocab are not cached (i.e., new for this input file)
        if _is_cached is False and len(self.lengths) == 0:
            for i in range(len(self.data)):
                _line_callback(self.data[i])

        # trim down the size of a newly created vocab
        if subword_path is None and vocab_size is not None:
            self.vocab.build_lookup(vocab_size)

        # -------------------------------------------------------------
        # save to cache if not already saved
        # -------------------------------------------------------------
        if _is_cached is False:
            print("Writing data to cache...")
            with open(_cfile, "wb") as f:
                pickle.dump((self.vocab, self.lengths), f)

        self.lengths = numpy.array(self.lengths)

    @staticmethod
    def init_vocab(vocab=None, subword_path=None, oovs=0):
        _is_prebuilt = True

        # use the given vocab
        if vocab is not None:
            _vocab = vocab

        # load vocab from disk
        elif vocab is None and subword_path is not None:
            _vocab = Vocab(oovs=oovs, subword=subword_path)
            _vocab.load_from_vocab_file(subword_path + ".vocab")

        # build vocab from the tokens in the dataset
        else:
            _vocab = Vocab(oovs=oovs, subword=subword_path)
            _vocab.reset()
            _is_prebuilt = False
        return _vocab, _is_prebuilt

    def dataitem(self, i):

        # tokenize sentence / text
        token_list = self.tokenize(self.data[i])

        # add special tokens such as <BOS> or <EOS>
        token_list = self.add_special_tokens(token_list)

        # vectorize the tokens
        vector = vectorize(token_list, self.vocab)
        return vector

    @staticmethod
    def _get_cache_key(input, vocab, tokenize, subword_path, vocab_size,
                       subsample):
        """

        Args:
            input:
            vocab:
            tokenize:
            oovs:

        Returns:

        """
        _hash = lambda x: hashlib.sha256(x.encode()).hexdigest()
        _cache_key = _hash(input) + str(os.stat(input).st_mtime)

        if vocab is not None:
            _cache_key += vocab.hash()

        if subsample is not None:
            _cache_key += str(subsample)

        _cache_key += _hash(inspect.getsource(tokenize))
        _cache_key += str(subword_path)
        _cache_key += str(vocab_size)
        return _hash(_cache_key)

    @staticmethod
    def space_tok(text):
        return text.rstrip().split()

    def add_special_tokens(self, tokens):
        tokens = tokens + [self.vocab.EOS]
        if self.sos:
            tokens = [self.vocab.SOS] + tokens
        if self.seq_len > 0:
            tokens = tokens[:self.seq_len]
        return tokens

    def properties(self):
        props = dict()
        props["file"] = os.path.basename(self.input)
        props["examples"] = len(self)
        props["vocab"] = len(self.vocab)
        props["tokens (unique)"] = len(self.vocab.vocab)
        props["tokens (total)"] = number_h(sum(self.lengths))

        # if len(self.data) < 4000000:
        #     unk_id = self.vocab.tok2id[self.vocab.UNK]
        #     _coverage = numpy.mean(
        #         [self.dataitem(i).count(unk_id) / len(self.dataitem(i))
        #          for i in range(len(self.data))])
        #     props["UNKs"] = f"{round((_coverage * 100), 2)} %"

        if hasattr(self, 'seq_len'):
            props["max length"] = self.seq_len
        if hasattr(self, 'bptt'):
            props["BPTT"] = self.bptt
        return props

    def __str__(self):
        return tabulate([self.properties()],
                        headers="keys", floatfmt=".4f", numalign="right")

    def truncate(self, n):
        self.data = self.data[:n]
        self.lengths = self.lengths[:n]


class SequenceDataset(BaseSequenceDataset):
    def __init__(self, *args, **kwargs):
        """
        Dataset for sentence-level Language Modeling.
        """

        # todo: hasty hack - fix after submission
        if kwargs.get("vocab") is not None and kwargs.get("vocab").is_gpt2:
            with open(args[0], encoding="utf-8") as f:
                self.vocab = kwargs.get("vocab")
                self.data = [self.vocab.gpt2_tok(line) for line in f]
                self.lengths = numpy.array([len(x) for x in self.data])
        else:
            super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.data)

    def get_sample(self, index):
        # sample = devectorize(self.data[index], self.vocab)
        # return sample
        raise NotImplementedError

    def __getitem__(self, index):
        tok_ids = self.dataitem(index)
        # tok_ids = self.data[index]
        inputs = tok_ids[:-1]
        targets = tok_ids[1:]

        return inputs, targets, len(inputs)


class TranslationDataset(Dataset):
    def __init__(self, src: SequenceDataset, trg: SequenceDataset,
                 verbose=True):
        self.src = src
        self.trg = trg

        if verbose:
            try:
                print(self)
                print()
            except:
                pass

    def __str__(self):
        return tabulate([self.src.properties(), self.trg.properties()],
                        headers="keys", floatfmt=".4f", numalign="right")

    def __len__(self):
        return len(self.src.data)

    def __getitem__(self, index):
        x_sos, x_eos, x_len = self.src[index]
        y_sos, y_eos, y_len = self.trg[index]
        return x_sos, x_eos, y_sos, y_eos, x_len, y_len
