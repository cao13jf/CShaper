'''Library for data sampling'''

import random
import torch
import numpy as np
from torch.utils.data.sampler import Sampler


#===========================================
#   Define iterators
#===========================================
#   Shuffle the list whenever the list is traversed
class RandomCycleIter(object):
    def __init__(self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0  # when reach the largest, reset the number
            random.shuffle(self.data_list)
        return self.data_list[self.i]

    next = __next__ #


#=============================================
#   Multisampler
#=============================================
class MultiSampler(object):
    def __init__(self, batch_sizes, sizes, num_samples=None, num_iters=None):
        '''
        :param batch_sizes: batch size in each iterator
        :param sizes:  size in each iterator
        :param num_samples: total number sample at each sample
        :param num_iters: iteration for sampling at each time
        '''
        self.batch_size = sum(batch_sizes)
        self.index_data = {}
        size, counter = 0, -1  # size: total number of data 'index_data', counter: which iterator this data comes from
        for i in range(self.batch_size):
            if i == size:  # when enough data from one iterator, change to another iterator
                counter += 1
                size += batch_sizes[i]
            self.index_data[i] = counter
        self.num_samples = num_samples or num_iters*self.batch_size or sum(sizes)
        self.data_iters = [RandomCycleIter(range(n)) for n in sizes]

    def __iter__(self):
        return multi_data_generator(
            self.data_iters, self.index_data,
            self.num_samples, self.batch_size
        )

    def __len__(self):
        return self.num_samples

def multi_data_generator(data_iters, index_data, n, size):
    i = 0
    while i < n:
        index = i % size
        d = index_data[index]
        yield d, next(data_iters[i])
        i += 1


#============================================================
#   Single sampler
#============================================================
class CycleSampler(Sampler):
    def __init__(self, size, num_samples=None, num_epochs=0):
        self.num_samples = num_samples or size*num_epochs
        self.data_iter = RandomCycleIter(range(size))

    def __iter__(self):
        return single_data_generator(self.data_iter, self.num_samples)

    def __len__(self):
        return self.num_samples

def single_data_generator(data_iter, n):
    i = 0
    while i < n:
        yield next(data_iter)
        i += 1


#===========================================================
#   Random sampler
#===========================================================
class RandomSampler(object):
    def __init__(self, data_list, seed=None):
        self.data_list = data_list
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        return iter(torch.randperm(len(self.data_list)).long())

    def __len__(self):
        return len(self.data_list)

    def get_state(self):
        return self.rng.get_state()

    def set_state(self, state):
        self.rng.set_state(state)