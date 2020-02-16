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