from typing import NamedTuple, Union
import abc
import copy
import uuid

import numpy as np
import torch

import torch_util
import visualization


class TensorBatch(NamedTuple):
    states: torch_util.FloatTensor
    actions: torch_util.LongTensor
    intermediate_returns: torch_util.FloatTensor
    bootstrap_states: torch_util.FloatTensor
    bootstrap_actions: torch_util.LongTensor
    bootstrap_weights: torch_util.FloatTensor


class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    intermediate_returns: np.ndarray
    bootstrap_states: np.ndarray
    bootstrap_actions: np.ndarray
    bootstrap_weights: np.ndarray

    def to_tensor(self, use_cuda: bool) -> TensorBatch:
        return TensorBatch(*torch_util.load_inputs(use_cuda, self.states, self.actions, self.intermediate_returns,
                                                   self.bootstrap_states, self.bootstrap_actions,
                                                   self.bootstrap_weights))


Batch_ = Union[Batch, TensorBatch]


class TemporalDifferenceBase(metaclass=abc.ABCMeta):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, target_update_rate: int):
        self._online_network = model
        self._target_network = copy.deepcopy(self._online_network)
        self._optimizer = optimizer
        self._target_update_rate = target_update_rate

        self._update_counter = 0
        self.name = str(uuid.uuid4())

    def update(self, batch: Batch_):
        loss = self._update(self.get_tensor_batch(batch))
        visualization.global_summary_writer.add_scalar(f'TD/TD loss ({self.name})', loss.data[0], self._update_counter)
        self._optimizer.zero_grad()
        loss.backward()
        # noinspection PyArgumentList
        self._optimizer.step()
        self._update_counter += 1
        if self._update_counter % self._target_update_rate == 0:
            self._target_network = copy.deepcopy(self._online_network)

    def get_tensor_batch(self, batch: Batch_) -> TensorBatch:
        if batch is TensorBatch:
            return batch
        else:
            return batch.to_tensor(torch_util.module_is_cuda(self._online_network))

    @abc.abstractmethod
    def _update(self, batch: TensorBatch) -> torch.autograd.Variable:
        pass
