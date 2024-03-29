import abc

import numpy as np


class DataAdapter(metaclass=abc.ABCMeta):
    def __init__(self, x, y=None, **kwargs):
        self.x = x
        self.y = y

    @abc.abstractmethod
    def get_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def batch_size(self):
        raise NotImplementedError


class TensorLikeDataAdapter(DataAdapter):
    def __init__(
        self, x, y, batch_size=None, epochs=1, steps=None, shuffle=False, **kwargs
    ):
        super(TensorLikeDataAdapter, self).__init__(x, y, **kwargs)

        self.num_samples = x.shape[0]
        if steps is None:
            steps = 32
        if batch_size is None:
            batch_size = self.num_samples // steps
        self._batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.shuffle = shuffle
        self.current_index = 0
        self.resetted = False
        self.num_batches = self.num_samples // batch_size

    def __len__(self):
        return self.num_batches

    def check_index_range(self, index_range, required_length):
        return index_range > required_length

    def get_dataset(self):
        return (self.x, self.y)

    def batch_size(self):
        return self._batch_size

    def get_batch(self):
        if self.shuffle:
            return self.get_with_shuffle()
        else:
            return self.get_without_shuffle()

    def get_with_shuffle(self):
        current_batch_indices = np.random.choice(self.num_samples, self._batch_size)
        current_batch_x = self.x[current_batch_indices]
        current_batch_y = self.y[current_batch_indices]
        return current_batch_x, current_batch_y

    def get_without_shuffle(self):
        current_batch_index = self.current_index * self._batch_size
        status = self.check_index_range(
            current_batch_index + self._batch_size, self.num_samples
        )
        if status:
            self.current_index = 0
            current_batch_index = 0
            self.resetted = True
        else:
            self.resetted = False
        current_batch_x = self.x[
            current_batch_index : current_batch_index + self._batch_size
        ]
        current_batch_y = self.y[
            current_batch_index : current_batch_index + self._batch_size
        ]
        self.current_index += 1
        return current_batch_x, current_batch_y
