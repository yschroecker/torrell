from typing import NamedTuple
import abc
import copy

import torch

import torch_util


class Batch(NamedTuple):
    states: torch_util.Tensor
    actions: torch_util.LongTensor
    intermediate_returns: torch_util.Tensor
    bootstrap_states: torch_util.Tensor
    bootstrap_actions: torch_util.LongTensor
    bootstrap_weights: torch_util.Tensor


class TemporalDifference(metaclass=abc.ABCMeta):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, target_update_rate: int):
        self._online_network = model
        self._target_network = copy.deepcopy(self._online_network)
        self._optimizer = optimizer
        self._target_update_rate = target_update_rate

        self._update_counter = 0

    def _update(self, loss: torch.autograd.Variable):
        self._optimizer.zero_grad()
        loss.backward()
        # noinspection PyArgumentList
        self._optimizer.step()
        self._update_counter += 1
        if self._update_counter % self._target_update_rate == 0:
            self._target_network = copy.deepcopy(self._online_network)

    @abc.abstractmethod
    def update(self, batch: Batch):
        pass
