import abc
from typing import Generic, TypeVar, Sequence, Optional, Type, Tuple

import collections

import numpy as np

import trainers.online_trainer
import trainers.data_collection
import trainers.ring_buffer
import data
import trainers.synchronous


class MemoryBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, batch_size: int) -> data.Batch[data.RLTransitionSequence]:
        pass

    @abc.abstractmethod
    def extend(self, batch: data.Batch[data.RLTransitionSequence]):
        pass

    @property
    @abc.abstractmethod
    def size(self) -> int:
        pass


class FIFOReplayMemory(MemoryBase):
    def __init__(self, memory_size: int):
        self._memory_size = memory_size
        self._buffer = collections.deque(maxlen=memory_size)
        self._discount_factor = None

    def sample(self, batch_size: int) -> data.Batch[data.RLTransitionSequence]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        sequences = [self._buffer[idx] for idx in indices]
        return data.Batch(sequences, self._discount_factor)

    def extend(self, batch: data.Batch[data.RLTransitionSequence]):
        self._discount_factor = batch.discount_factor
        self._buffer.extend(batch.sequences)

    @property
    def size(self) -> int:
        return len(self._buffer)


class MixedBatchExperienceReplay:
    def __init__(self, trainer: trainers.data_collection.Trainer, memory_size: int,
                 batch_size: int, initial_population: int, sample_batch_size: int=1):
        self._trainer = trainer
        self._batch_size = batch_size
        self._memory = FIFOReplayMemory(memory_size)
        self._initial_population = initial_population
        self._sample_batch_size = sample_batch_size

    def collect_transitions(self) -> Optional[data.Batch[data.RLTransitionSequence]]:
        sample_batch = trainers.data_collection.collect_batch(self._trainer, self._sample_batch_size, 1)
        self._memory.extend(sample_batch)

        if self._memory.size >= self._initial_population:
            memory_batch = self._memory.sample(self._batch_size)
            return memory_batch
        return None
