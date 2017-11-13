import abc
from typing import Generic, TypeVar, Sequence, Optional, Type, Tuple

import collections

import numpy as np
import torch
import torch.nn.functional as f

import trainers.online_trainer
import trainers.ring_buffer
from environments.environment import Environment
import data
import trainers.synchronous
import policies.policy
import torch_util
import visualization


class MemoryBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, batch_size: int) -> data.Batch[data.RLTransitionSequence]:
        pass

    @abc.abstractmethod
    def extend(self, batch: data.Batch[data.RLTransitionSequence]):
        pass

    @abc.abstractmethod
    @property
    def size(self) -> int:
        pass


class FIFOReplayMemory(MemoryBase):
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
    def size(self) -> int:
        return len(self._buffer)


class MemoryWithPolicy(MemoryBase):
    def __init__(self, memory: MemoryBase, policy: policies.policy.PolicyModel, optimizer: torch.optim.Optimizer):
        self._memory = memory
        self._memory_policy = policy
        self._optimizer = optimizer

    def _update_policy(self, memory_batch: data.Batch[data.RLTransitionSequence]):
        X = np.array(memory_batch.states(), dtype=np.float32)
        y = memory_batch.actions()
        y_tensor = torch_util.load_input(self._memory_policy.is_cuda, y)
        X_tensor = torch_util.load_input(self._memory_policy.is_cuda, X)
        X_var = torch.autograd.Variable(X_tensor, requires_grad=False)
        y_var = torch.autograd.Variable(y_tensor, requires_grad=False)
        loss = self._memory_policy.log_probability(X_var, y_var).mean()

        loss_tensor = loss.data
        if self._memory_policy.is_cuda:
            loss_tensor = loss_tensor.cpu()

        visualization.global_summary_writer.add_scalar('m loss', loss_tensor.numpy())
        return loss

    def sample(self, batch_size: int) -> data.Batch[data.RLTransitionSequence]:
        batch = self._memory.sample(batch_size)
        self._update_policy(batch)
        return batch

    def extend(self, batch: data.Batch[data.RLTransitionSequence]):
        self._memory.extend(batch)

    @property
    def size(self) -> int:
        return self._memory.size


class MixedBatchExperienceReplay(trainers.online_trainer.DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: trainers.online_trainer.TrainerConfig, memory_size: int,
                 batch_size: int, initial_population: int, sample_batch_size: int=1):
        super().__init__(env, trainer_config)
        self._batch_size = batch_size
        self._memory = FIFOReplayMemory(trainer_config.state_dim, trainer_config.action_dim,
                                        trainer_config.policy_model.action_type, memory_size)
        self._initial_population = initial_population
        self._sample_batch_size = sample_batch_size

    def collect_transitions(self, _) -> Optional[data.Batch[data.RLTransitionSequence]]:
        sample_batch = self._collect_transitions(self._sample_batch_size)
        self._memory.extend(sample_batch)

        if self._memory.size >= self._initial_population:
            memory_batch = self._memory.sample(self._batch_size)
            return memory_batch
        return None


class AlternatingBatchExperienceReplay(trainers.synchronous.SynchronizedDiscreteNstepTrainer):
    def __init__(self, envs: Sequence[Environment[int]], trainer_config: trainers.online_trainer.TrainerConfig,
                 look_ahead: int, batch_size: int, memory_iterations_ratio: int, burn_in: int,
                 memory: MemoryBase):
        super().__init__(envs, trainer_config, look_ahead, batch_size)
        self._memory = memory
        self._memory_iteration_ratio = memory_iterations_ratio
        self._burn_in = burn_in

    def collect_transitions(self, iteration: int) -> Optional[data.Batch[data.RLTransitionSequence]]:
        if iteration % self._memory_iteration_ratio == 0 and iteration >= self._burn_in:
            batch = super().collect_transitions(iteration)
            self._memory.extend(batch)
            return batch
        else:
            return self._memory.sample(self._batch_size)

