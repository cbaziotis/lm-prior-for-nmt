class MultiDataLoader(object):
    def __init__(self, loaders, strategy="min", ratios=None, names=None):
        """
        Wrapper for combining multiple dataloaders. It returns batches from the
        given dataloaders, based on the a given strategy.
        The dataloaders can be from different datasets, with different sizes,
        structure etc. The wrapper makes no assumptions.

        A useful use case might be semi-supervised learning, where for instance
        in machine translation you might want to combine batches from some
        parallel corpora (translation) with batches from some
        monolingual corpora (LM-Autoencoder).

        Args:
            loaders:
            strategy: spread, modulo, cycle, beginning

                min: this iterator will terminate, when the shortest loader
                    (the one with the minimum number of batches) terminates.
                cycle: this iterator will terminate, when the longest loader
                    (the one with the maximum number of batches) terminates.
                    If one of the shorter ones terminates, it will be restarted.
                ratio: iterates over the dataloaders based on a list of rations.
                    the longest loader should have a ratio of 1.

                    Example: suppose you passed 5 iterators,  with
                        lengths: [13, 2, 2, 2, 2]
                        ratios:  [1, 2, 2, 3, 3]

                        Then a full iteration over all the dataloaders, will
                        yield the following batch (iter_id):
                        0, 1, 2, 3, 4, 0, 0, 1, 2, 0, 3, 4,
                        0, 1, 2, 0, 0, 1, 2, 3, 4, 0, 0, 1,
                        2, 0, 3, 4, 0, 1, 2, 0, 0, 1, 2, 3, 4
        """
        self.loaders = loaders
        self.strategy = strategy
        self.sizes = [len(loader) for loader in self.loaders]
        self.ratios = ratios
        self.iter_id = None

        if names is None:
            self.names = [i for i in self.loaders]
        else:
            self.names = names

    def __iter__(self):
        return _MultiDataLoaderIter(self)

    def get_current_loader(self):
        return self.names[self.iter_id]

    def __len__(self):
        if self.strategy == "cycle":
            return max(self.sizes) * len(self.sizes)
        elif self.strategy == "ratio":
            raise NotImplementedError
        else:
            return sum(self.sizes)


class _MultiDataLoaderIter(object):
    def __init__(self, loader: MultiDataLoader):
        self.loader = loader
        self.loaders = loader.loaders
        self.strategy = loader.strategy
        self.ratios = loader.ratios
        self.iterators = [iter(loader) for loader in loader.loaders]
        self.sizes = [len(loader) for loader in loader.loaders]
        self.names = loader.names

        self.step = 0

        if self.strategy == "ratio":
            self.steps = [0 for _ in range(len(self.loaders))]

    def __iter__(self):
        return self

    def _reset_if_not_max(self, iter_id):
        # if the longest iterator ends, stop
        if self.sizes[iter_id] == max(self.sizes):
            raise StopIteration
        # else reset the iterator and continue
        else:
            self.iterators[iter_id] = iter(self.loaders[iter_id])
            return next(self.iterators[iter_id])

    def __next__(self):

        if self.strategy == "min":
            iter_id = self.step % len(self.iterators)
            batch = next(self.iterators[iter_id])
            self.step += 1
            self.loader.iter_id = iter_id
            return batch

        elif self.strategy == "cycle":
            iter_id = self.step % len(self.iterators)
            batch = next(self.iterators[iter_id], None)

            if batch is None:
                batch = self._reset_if_not_max(iter_id)

            self.step += 1
            self.loader.iter_id = iter_id
            return batch

        elif self.strategy == "ratio":
            iter_id = self.step % len(self.iterators)

            steps = self.steps[iter_id]
            ratio = self.ratios[iter_id]

            self.step += 1
            self.steps[iter_id] += 1

            if steps % ratio == 0:
                batch = next(self.iterators[iter_id], None)

                if batch is None:
                    batch = self._reset_if_not_max(iter_id)

            else:
                batch = next(self)

            self.loader.iter_id = iter_id
            return batch

        else:
            raise ValueError
