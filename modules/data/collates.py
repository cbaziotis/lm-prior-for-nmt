import torch
from torch.nn.utils.rnn import pad_sequence


class SeqCollate:
    """
    Base Class.
    A variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, *args):
        pass
        # self.sort = sort
        # self.batch_first = batch_first

    def pad_samples(self, samples):
        return pad_sequence([torch.LongTensor(x) for x in samples],
                            batch_first=True)

    def _collate(self, *args):
        raise NotImplementedError

    def __call__(self, batch):
        batch = list(zip(*batch))
        return self._collate(*batch)


class LMCollate(SeqCollate):
    def __init__(self, bidirectional=False, *args):
        super().__init__(*args)
        self.bidirectional = bidirectional

    def _collate(self, inputs, targets, lengths):

        inputs_fwd = self.pad_samples(inputs)
        targets_fwd = self.pad_samples(targets)
        lengths = torch.LongTensor(lengths)

        if self.bidirectional:
            # must flip not only order of elements but also targets <-> inputs !
            inputs_bwd = [x[::-1] for x in targets]
            targets_bwd = [x[::-1] for x in inputs]

            inputs_bwd = self.pad_samples(inputs_bwd)
            targets_bwd = self.pad_samples(targets_bwd)

            return inputs_fwd, targets_fwd, inputs_bwd, targets_bwd, lengths
        else:
            return inputs_fwd, targets_fwd, lengths


class CondLMCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inputs, targets, attributes, lengths):
        inputs = self.pad_samples(inputs)
        targets = self.pad_samples(targets)
        attributes = self.pad_samples(attributes)
        lengths = torch.LongTensor(lengths)
        return inputs, targets, attributes, lengths


class Seq2SeqBatch:
    def __init__(self, src_inp, src_out, trg_inp, trg_out, src_len, trg_len):
        self.src_inp = self.pad_samples(src_inp)
        self.src_out = self.pad_samples(src_out)
        self.trg_inp = self.pad_samples(trg_inp)
        self.trg_out = self.pad_samples(trg_out)
        self.src_len = torch.LongTensor(src_len)
        self.trg_len = torch.LongTensor(trg_len)

    def pad_samples(self, samples):
        return pad_sequence([torch.LongTensor(x) for x in samples], True)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.src_inp = self.src_inp.pin_memory()
        self.src_out = self.src_out.pin_memory()
        self.trg_inp = self.trg_inp.pin_memory()
        self.trg_out = self.trg_out.pin_memory()
        self.src_len = self.src_len.pin_memory()
        self.trg_len = self.trg_len.pin_memory()
        return self.src_inp, self.src_out, self.src_len, self.trg_inp, self.trg_out, self.trg_len


class Seq2SeqCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, src_inp, src_out, trg_inp, trg_out, src_len, trg_len):
        src_inp = self.pad_samples(src_inp)
        src_out = self.pad_samples(src_out)
        trg_inp = self.pad_samples(trg_inp)
        trg_out = self.pad_samples(trg_out)

        src_len = torch.LongTensor(src_len)
        trg_len = torch.LongTensor(trg_len)

        # lengths_sorted, sorted_i = src_len.sort(descending=True)
        # src_inp = src_inp[sorted_i]
        # src_out = src_out[sorted_i]
        # trg_inp = trg_inp[sorted_i]
        # trg_out = trg_out[sorted_i]
        # src_len = src_len[sorted_i]
        # trg_len = trg_len[sorted_i]
        # return Seq2SeqBatch(*list(zip(*batch)))
        return src_inp, src_out, src_len, trg_inp, trg_out, trg_len


class Seq2SeqOOVCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inp_src, out_src, inp_trg, out_trg, len_src, len_trg,
                 oov_map):
        inp_src = self.pad_samples(inp_src)
        out_src = self.pad_samples(out_src)
        inp_trg = self.pad_samples(inp_trg)
        out_trg = self.pad_samples(out_trg)

        len_src = torch.LongTensor(len_src)
        len_trg = torch.LongTensor(len_trg)

        return inp_src, out_src, len_src, inp_trg, out_trg, len_trg, oov_map
