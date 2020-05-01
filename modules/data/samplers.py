import math

import numpy
import torch
from torch.utils.data import Sampler


class BPTTSampler(Sampler):
    """
    Samples elements per chunk. Suitable for Language Models.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, size, batch):
        """
        Define how to construct batches

        Given a list of sequences, organize the sequences in each batch
        in such a way, so that each RNN gets the proper (next) sequence.

        For example, given the following sequence and with batch=2:
        ┌ a b c d e ┐
        │ f g h i j │
        │ k l m n o │
        │ p q r s t │
        │ u v w x y │
        └ z - - - - ┘

        the batches will be:
        ┌ a b c d e ┐    ┌ f g h i j ┐    ┌ k l m n o ┐
        └ p q r s t ┘    └ u v w x y ┘    └ z - - - - ┘

        Args:
            size (int): number of sequences
            batch (int): batch size
        """
        self.size = size
        self.batch = batch

        # split the corpus in chunks of size `corpus_seqs / batch_size`
        self.chunks = numpy.array_split(numpy.arange(self.size), batch)

    def get_batch(self, index):
        """
        Fill each batch with the i-th sequence from each chunk.
        If the batch size does not evenly divides the chunks,
        then some chunks will have one less sequence, so the last batch
        will have fewer samples.
        Args:
            index (int):

        Returns:

        """
        batch = []
        for chunk in self.chunks:
            if index < chunk.size:
                batch.append(chunk[index])
        return batch

    def batches(self):
        for i in range(self.chunks[0].size):
            yield self.get_batch(i)

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return self.size


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


class SortedSampler(Sampler):
    """
    Defines a strategy for drawing samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, descending=False):
        self.lengths = lengths
        self.desc = descending

    def __iter__(self):

        if self.desc:
            return iter(numpy.flip(numpy.array(self.lengths).argsort(), 0))
        else:
            return iter(numpy.array(self.lengths).argsort())

    def __len__(self):
        return len(self.lengths)


class TokenBatchSampler(Sampler):
    """
    Defines a strategy for drawing batches of samples from the dataset,

    Important: Since we are not shuffling the data and it's inevitable
    that the sentences in the batch will be uneven in terms of their length,
    instead of summing the lengths of the sentences in each batch to compute
    the threshold for creating a new batch, we use as the maximum length
    in the batch times the sentences in the batch

    Example:
        [
            [x, x, x, x, x, x, x, x, x, x, x, x, x, x ]
            [x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
                        ....
            [x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0 ]
        ]
    """

    def __init__(self, lengths, batch_tokens):
        self.batches = []

        accumulator = 0

        batch = []
        for index, length in enumerate(lengths):

            if len(batch) == 0:
                accumulator = length
            else:
                accumulator = max(accumulator, length)

            if accumulator * len(batch) < batch_tokens:
                batch.append(index)
            else:
                # insert new batch
                self.batches.append(batch)

                # create new batches
                batch = [index]
                accumulator = length

        if len(batch) > 0:
            self.batches.append(batch)

        assert not any([sum(lengths[y] for y in x) > batch_tokens
                        for x in self.batches])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class BucketBatchSampler(Sampler):
    """
    Defines a strategy for drawing batches of samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, batch_size,
                 shuffle=False, even=False, drop_last=False, reverse=False):
        sorted_indices = numpy.array(lengths).argsort()
        num_sections = math.ceil(len(lengths) / batch_size)
        if even:
            self.batches = list(divide_chunks(sorted_indices, batch_size))
        else:
            self.batches = numpy.array_split(sorted_indices, num_sections)

        if reverse:
            self.batches = list(reversed(self.batches))

        if drop_last:
            del self.batches[-1]

        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(self.batches[i]
                        for i in torch.randperm(len(self.batches)))
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class BucketTokensSampler(Sampler):
    """
    Defines a strategy for drawing batches of samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    The batches will be contructed based on the total number of tokens.
    """

    def __init__(self, lengths, batch_tokens,
                 shuffle=False, drop_last=False):

        self.shuffle = shuffle
        self.lengths = lengths
        self.batch_tokens = batch_tokens
        self.drop_last = drop_last

        self.batches, self.reverse_ids = self.get_batches()

    def get_batches(self):

        if self.shuffle:
            # this ensures shuffling inside batches
            s = numpy.random.randint(-1, 1, len(self.lengths))
            sorted_indices = numpy.array(self.lengths + s).argsort()
        else:
            sorted_indices = numpy.array(self.lengths).argsort()

        # self.reverse_ids = sorted_indices[::-1]
        reverse_ids = numpy.array(sorted_indices).argsort()
        batches = []
        batch = []
        accumulator = 0

        for index in sorted_indices:
            accumulator += self.lengths[index]

            if accumulator < self.batch_tokens:
                batch.append(index)
            else:
                # insert new batch
                batches.append(batch)

                # create new batches
                batch = [index]
                accumulator = self.lengths[index]

        if self.drop_last:
            del batches[-1]
        elif len(batch) > 0:
            batches.append(batch)

        assert not any([sum(self.lengths[y] for y in x) > self.batch_tokens
                        for x in batches])

        return batches, reverse_ids

    def __iter__(self):
        if self.shuffle:
            # get fresh order of batches
            self.batches, self.reverse_ids = self.get_batches()

            return iter(self.batches[i]
                        for i in torch.randperm(len(self.batches)))
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)
