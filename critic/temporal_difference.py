from typing import NamedTuple
import abc
import copy

import numpy as np
import torch

import torch_util


class Batch(NamedTuple):
    states: torch_util.Tensor
    actions: torch_util.LongTensor
    intermediate_returns: torch_util.Tensor
    bootstrap_states: torch_util.Tensor
    bootstrap_actions: torch_util.LongTensor
    bootstrap_weights: torch_util.Tensor

    @classmethod
    def from_numpy(cls, states: np.ndarray, actions: np.ndarray, intermediate_returns: np.ndarray,
                   bootstrap_states: np.ndarray, bootstrap_actions: np.ndarray, bootstrap_weights: np.ndarray) -> \
            NamedTuple:
        return Batch(
            states=torch.from_numpy(np.array(states, dtype=np.float32)).type(torch_util.Tensor),
            actions=torch.from_numpy(np.array(actions, dtype=np.int32)).type(torch_util.LongTensor),
            intermediate_returns=torch.from_numpy(np.array(intermediate_returns, dtype=np.float32)).type(
                torch_util.Tensor),
            bootstrap_weights=torch.from_numpy(bootstrap_weights).type(torch_util.Tensor),
            bootstrap_states=torch.from_numpy(np.array(bootstrap_states, dtype=np.float32)).type(torch_util.Tensor),
            bootstrap_actions=torch.from_numpy(np.array(bootstrap_actions, dtype=np.int32)).type(torch_util.LongTensor)
        )


class TemporalDifferenceBase(metaclass=abc.ABCMeta):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, target_update_rate: int):
        self._online_network = model
        self._target_network = copy.deepcopy(self._online_network)
        self._optimizer = optimizer
        self._target_update_rate = target_update_rate

        self._update_counter = 0
        self.name = ""

    def _update(self, loss: torch.autograd.Variable):
        torch_util.global_summary_writer.add_scalar(f'TD/TD loss ({self.name})', loss.data[0], self._update_counter)
        self._optimizer.zero_grad()
        loss.backward()
        for name, parameter in self._online_network.named_parameters():
            torch_util.global_summary_writer.add_scalar(f'{name} ({self.name})', parameter.grad.data,
                                                        self._update_counter)
        torch_util.global_summary_writer.add_scalar(f'TD/TD loss ({self.name})', loss.data[0], self._update_counter)
        # noinspection PyArgumentList
        self._optimizer.step()
        self._update_counter += 1
        if self._update_counter % self._target_update_rate == 0:
            self._target_network = copy.deepcopy(self._online_network)

    @abc.abstractmethod
    def update(self, batch: Batch):
        pass
