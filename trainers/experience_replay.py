from typing import Generic, TypeVar, Sequence, Optional, Type, Tuple

import collections

import numpy as np
import tqdm

import trainers.online_trainer
import trainers.ring_buffer
from environments.environment import Environment
import data


ActionT = TypeVar('ActionT')
SampleType = Tuple[np.ndarray, ActionT, float, float, np.ndarray, ActionT]


class FIFOReplayMemory(Generic[ActionT]):
    def __init__(self, state_dim: int, action_dim: int, action_type: Type[np.dtype], memory_size: int):
        self._memory_size = memory_size
        self._buffer = collections.deque(maxlen=memory_size)

    def sample(self, batch_size: int) -> data.Batch[data.RLTransitionSequence]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        sequences = [self._buffer[idx] for idx in indices]
        return data.Batch(sequences)

    def extend(self, batch: data.Batch[data.RLTransitionSequence]):
        self._buffer.extend(batch.sequences)

    @property
    def size(self):
        return len(self._buffer)


class DiscreteExperienceReplay(trainers.online_trainer.DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: trainers.online_trainer.TrainerConfig, memory_size: int,
                 batch_size: int, initial_population: int, sample_batch_size: int=1):
        super().__init__(env, trainer_config)
        self._batch_size = batch_size
        self._memory = FIFOReplayMemory(trainer_config.state_dim, trainer_config.action_dim,
                                        trainer_config.policy_model.action_type, memory_size)
        self._initial_population = initial_population
        self._sample_batch_size = sample_batch_size

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            self._iterate(iteration, trange)

    def _iterate(self, iteration: int, trange: tqdm.tqdm):
        sample_batch = self.collect_transitions(self._sample_batch_size)
        self._memory.extend(sample_batch)

        if self._memory.size >= self._initial_population:
            memory_batch = self._memory.sample(self._batch_size)
            trange.set_description(self.do_train(iteration, memory_batch))
