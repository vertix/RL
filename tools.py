import os
import random
import time

import numpy as np
import tensorflow as tf


class ExperienceBuffer(object):
    """Simple experience buffer"""

    def __init__(self, buffer_size=1 << 16):
        self.ss, self.aa, self.rr, self.ss1, self.gg = None, None, None, None, None
        self.buffer_size = buffer_size
        self.inserted = 0
        self.index = []

    def add(self, s, a, r, s1, gamma, unused_weight):
        if self.ss is None:
            state_size = s.shape[1]
            self.ss = np.zeros(
                (state_size, self.buffer_size), dtype=np.float32)
            self.aa = np.zeros(self.buffer_size, dtype=np.int16)
            self.ss1 = np.zeros(
                (state_size, self.buffer_size), dtype=np.float32)
            self.rr = np.zeros(self.buffer_size, dtype=np.float32)
            self.gg = np.zeros(self.buffer_size, dtype=np.float32)

        indexes = []
        for _ in a:
            cur_index = self.inserted % self.buffer_size
            self.inserted += 1
            indexes.append(cur_index)

        self.ss[:, indexes] = s.transpose()
        self.aa[indexes] = a
        self.rr[indexes] = r
        self.ss1[:, indexes] = s1.transpose()
        self.gg[indexes] = gamma

        if len(self.index) < self.buffer_size:
            self.index.append(self.inserted)
        self.inserted += 1

    @property
    def state_size(self):
        return None if self.ss is None else self.ss.shape[0]

    def tree_update(self, buffer_index, new_weight):
        pass

    def sample(self, size):
        if size > self.inserted:
            return None, None, None, None, None

        indexes = np.random.choice(min(self.buffer_size, self.inserted), size)
        return (indexes, np.transpose(self.ss[:, indexes]), self.aa[indexes], self.rr[indexes],
                np.transpose(self.ss1[:, indexes]), self.gg[indexes], np.ones(len(indexes)))


class WeightedExperienceBuffer(object):

    def __init__(self, alpha, beta, max_weight, buffer_size=1 << 16):
        self.ss, self.aa, self.rr, self.ss1, self.gg = None, None, None, None, None
        self.buffer_size = buffer_size
        self.inserted = 0
        self.tree_size = buffer_size << 1
        # root is 1
        self.weight_sums = np.zeros(self.tree_size)
        self.weight_min = np.ones(self.tree_size) * (max_weight ** alpha)
        self.max_weight = max_weight
        self.alpha = alpha
        self.beta = beta

    def update_up(self, index):
        self.weight_sums[index] = self.weight_sums[
            index << 1] + self.weight_sums[(index << 1) + 1]
        self.weight_min[index] = min(
            self.weight_min[index << 1], self.weight_min[(index << 1) + 1])
        if index > 1:
            self.update_up(index >> 1)

    def index_in_tree(self, buffer_index):
        return buffer_index + self.buffer_size

    def index_in_buffer(self, tree_index):
        return tree_index - self.buffer_size

    def tree_update(self, buffer_index, new_weight):
        index = self.index_in_tree(buffer_index)
        new_weight = min(new_weight + 0.01, self.max_weight) ** self.alpha

        self.weight_sums[index] = new_weight
        self.weight_min[index] = new_weight
        self.update_up(index >> 1)

    def add(self, s, a, r, s1, gamma, weight):
        if self.ss is None:
            # Initialize
            state_size = s.shape[1]
            self.ss = np.zeros(
                (state_size, self.buffer_size), dtype=np.float32)
            self.aa = np.zeros(self.buffer_size, dtype=np.int16)
            self.ss1 = np.zeros(
                (state_size, self.buffer_size), dtype=np.float32)
            self.rr = np.zeros(self.buffer_size, dtype=np.float32)
            self.gg = np.zeros(self.buffer_size, dtype=np.float32)

        indexes = []
        for _ in a:
            cur_index = self.inserted % self.buffer_size
            self.inserted += 1
            indexes.append(cur_index)

        self.ss[:, indexes] = s.transpose()
        self.aa[indexes] = a
        self.rr[indexes] = r
        self.ss1[:, indexes] = s1.transpose()
        self.gg[indexes] = gamma

        for idx in indexes:
            self.tree_update(idx, weight)

    @property
    def state_size(self):
        return None if self.ss is None else self.ss.shape[0]

    def find_sum(self, node, sum):
        if node >= self.buffer_size:
            return self.index_in_buffer(node)
        left = node << 1
        left_sum = self.weight_sums[left]
        if sum < left_sum:
            return self.find_sum(left, sum)
        else:
            return self.find_sum(left + 1, sum - left_sum)

    def sample_indexes(self, size):
        total_weight = self.weight_sums[1]
        indexes = np.zeros(size, dtype=np.int32)
        for i in xrange(size):
            search = np.random.random() * total_weight
            indexes[i] = self.find_sum(1, search)
        return indexes

    def sample(self, size):
        if size > self.inserted:
            return None, None, None, None, None, None, None

        indexes = self.sample_indexes(size)
        max_w = (self.weight_min[1] / self.weight_sums[1]) ** -self.beta
        w = (self.weight_sums[self.index_in_tree(indexes)] /
             self.weight_sums[1]) ** -self.beta

        return (indexes,
                np.transpose(self.ss[:, indexes]), self.aa[
                    indexes], self.rr[indexes],
                np.transpose(self.ss1[:, indexes]), self.gg[indexes],
                w / max_w)


def HuberLoss(tensor, boundary):
    abs_x = tf.abs(tensor)
    delta = boundary
    quad = tf.minimum(abs_x, delta)
    lin = (abs_x - quad)
    return 0.5 * quad**2 + delta * lin


def ClipGradient(grads, clip):
    if clip > 0:
        gg = [g for g, _ in grads]
        vv = [v for _, v in grads]
        global_norm = tf.global_norm(gg)
        tf.summary.scalar('Scalars/Grad_norm', global_norm)
        grads = zip(tf.clip_by_global_norm(gg, clip, global_norm)[0], vv)
    return grads

