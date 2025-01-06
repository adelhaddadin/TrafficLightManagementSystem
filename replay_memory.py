import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx
        right = left + 1
        if left >= 2 * self.capacity:
            return idx
        if self.tree[left] >= s:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[1]

    def add(self, p, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def _get_priority(self, td_error):
        # Slightly larger epsilon inside to avoid zero priority
        return (abs(td_error) + 1e-5) ** self.alpha

    def add(self, td_error, sample):
        p = self._get_priority(td_error)
        self.tree.add(p, sample)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            s = random.random() * segment + i * segment
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        total = self.tree.total()
        sampling_probs = np.array(priorities) / total
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()

        return idxs, batch, weights

    def update(self, idx, td_error):
        p = self._get_priority(td_error)
        self.tree.update(idx, p)
