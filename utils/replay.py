import numpy as np
from typing import Any, Optional, Sequence


class Replay(object):

  def __init__(self, capacity: int, nrng):
    self._data = None  # type: Optional[Sequence[np.ndarray]]
    self._capacity = capacity
    self._num_added = 0
    self._nrng = nrng

  def add(self, items: Sequence[Any]):
    """Adds a single sequence of items to the replay.

    Args:
      items: Sequence of items to add. Does not handle batched or nested items.
    """
    if self._data is None:
      self._preallocate(items)

    for slot, item in zip(self._data, items):
      slot[self._num_added % self._capacity] = item

    self._num_added += 1

  def peekFirst(self):
    return self._data[0]

  def peekLast(self):
    return self._data[-1]

  def sample_priority(self, n):
    data = [slot[range(self.size)] for slot in self._data]
    priorities = np.array(np.power(data[0][:, 0] + 1e-12, self._alpha),
                         dtype=np.float64)

    priority_probs = np.divide(priorities, np.sum(priorities),
              out=np.zeros_like(priorities),
              where=np.sum(priorities) != 0)
    w = np.where(self.size * priority_probs == 0, np.zeros_like(self.size * priority_probs),
                 np.power(self.size * priority_probs, -self._beta))
    w = np.divide(w, np.max(w),
              out=np.zeros_like(w),
              where=np.max(w) != 0)
    try:
      sampled = self._nrng.multinomial(n=n, pvals=priority_probs)
    except:
      print("dad")
    sampled_indices = np.where(sampled > 0)[0]
    more_indices = []
    multiple_sampled_indices = np.where(sampled > 1)[0]
    how_many_times_more = sampled[multiple_sampled_indices] - 1
    for i, how_many in zip(multiple_sampled_indices, how_many_times_more):
      for _ in range(how_many):
        more_indices.append(i)

    sampled_indices = np.concatenate([sampled_indices, np.array(more_indices, dtype=np.int32)])
    return w[sampled_indices], [slot[sampled_indices] for slot in data]

  def change_priority(self, indices, priorities):
      for i, index in enumerate(indices):
        for j, slot in enumerate(self._data):
          self._data[j][index] = priorities[i]

  def peek_n_priority(self, n):
    sorted_indices = np.flip(np.argsort(self._data[0])[:, 0], axis=0)
    peek_n = []
    for k in range(len(self._data)):
      peek_n.append(self._data[k][sorted_indices][:n])

    return peek_n

  def sample(self, size: int) -> Sequence[np.ndarray]:
    """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
    indices = self._nrng.randint(self.size, size=size)
    return [slot[indices] for slot in self._data]

  def reset(self,):
    """Resets the replay."""
    self._data = None

  @property
  def size(self) -> int:
    return min(self._capacity, self._num_added)

  @property
  def fraction_filled(self) -> float:
    return self.size / self._capacity

  def _preallocate(self, items: Sequence[Any]):
    """Assume flat structure of items."""
    as_array = []
    for item in items:
      if item is None:
        raise ValueError('Cannot store `None` objects in replay.')
      as_array.append(np.asarray(item))

    self._data = [np.zeros(dtype=x.dtype, shape=(self._capacity,) + x.shape)
                  for x in as_array]

  def __repr__(self):
    return 'Replay: size={}, capacity={}, num_added={}'.format(
        self.size, self._capacity, self._num_added)