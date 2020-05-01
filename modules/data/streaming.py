import os
import pickle
from collections import Iterable
from typing import Callable

from tqdm import tqdm


class DatasetCache:
    """
    Adapted from https://github.com/Yelrose/linecache_light

    DatasetCache caches the line position of a file in the memory.
    Every time it access a line, it will seek to the corresponding position
    and readline(). The first time is slow, as the cache file has to be built

    Usage:
        linecache = DatasetCache('a.txt', cache_suffix='.cache')
        num_lines = len(linecache)
        line_0 = linecache[0]
        line_100 = linecache[100]

    """

    def __init__(self, filename, cache_suffix='.cache',
                 callback: Callable = None,
                 subsample: int = 0):
        self.filename = filename
        self.subsample = subsample
        if self.subsample > 0:
            cache_suffix = f".subsample_{subsample}" + cache_suffix
        if os.path.exists(self.filename + cache_suffix):

            self.st_mtime, self.line_offsets = pickle.load(
                open(self.filename + cache_suffix, 'rb'))
            self.num_lines = len(self.line_offsets)

            if self.st_mtime != os.stat(self.filename).st_mtime:
                print('The cache file is out-of-date')
                self._build_offset_dict(cache_suffix, callback)
        else:
            self._build_offset_dict(cache_suffix, callback)

        self.fhandle = None

    def _build_offset_dict(self, cache_suffix, callback):
        stat_info = os.stat(self.filename)
        self.st_mtime = stat_info.st_mtime

        desc = f"Caching lines to {self.filename + cache_suffix}"
        print(desc)
        pbar = tqdm(total=os.path.getsize(self.filename))

        with open(self.filename, 'rb') as f:
            self.line_offsets = []
            while True:
                offset_pos = f.tell()
                line = f.readline()

                if not line:
                    break

                if 0 < self.subsample <= len(self.line_offsets):
                    break

                self.line_offsets.append(offset_pos)

                if offset_pos > 0:
                    pbar.update(self.line_offsets[-1] - self.line_offsets[-2])

                if callback is not None and callable(callback):
                    callback(line.decode("utf-8"))

            pickle.dump((self.st_mtime, self.line_offsets),
                        open(self.filename + cache_suffix, 'wb'))
            self.num_lines = len(self.line_offsets)

        pbar.close()

    def get_handle(self):
        # self.fhandle = open(self.filename, 'rb', os.O_RDONLY | os.O_NONBLOCK)
        return open(self.filename, 'rb', os.O_RDONLY | os.O_NONBLOCK)

    def set_handle(self, handle):
        self.fhandle = handle

    def reset(self):
        if self.fhandle is None:
            self.fhandle.close()

    def __getitem__(self, line_no):
        if self.fhandle is None:
            self.fhandle = open(self.filename, 'rb',
                                os.O_RDONLY | os.O_NONBLOCK)

        f = self.fhandle

        # with open(self.filename, 'rb', os.O_RDONLY | os.O_NONBLOCK) as f:
        if isinstance(line_no, slice):
            return [self[ii] for ii in range(*line_no.indices(len(self)))]
        elif isinstance(line_no, Iterable):
            return [self[ii] for ii in line_no]
        else:
            if line_no >= self.num_lines:
                raise IndexError(
                    "Out of index: line_no:%s  num_lines: %s" % (
                        line_no, self.num_lines))

            f.seek(self.line_offsets[line_no])
            return f.readline().decode("utf-8")

    def __len__(self):
        return self.num_lines
