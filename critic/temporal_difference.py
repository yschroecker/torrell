from typing import Optional, NamedTuple, Union, Sequence
import abc
import copy

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
    importance_weights: torch_util.FloatTensor


class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    intermediate_returns: np.ndarray
    bootstrap_states: np.ndarray
    bootstrap_actions: np.ndarray
    bootstrap_weights: np.ndarray
    importance_weights: Optional[np.ndarray] = None

    def to_tensor(self, use_cuda: bool) -> TensorBatch:
        return TensorBatch(*torch_util.load_inputs(use_cuda, self.states, self.actions, self.intermediate_returns,
                                                   self.bootstrap_states, self.bootstrap_actions,
                                                   self.bootstrap_weights),
                            torch_util.load_input(
                                use_cuda, np.ones_like(self.intermediate_returns)
                                if self.importance_weights is None else self.importance_weights))

Batch_ = Union[Batch, TensorBatch]


class TemporalDifferenceBase(metaclass=abc.ABCMeta):
    def __init__(self, model: torch.nn.Module, target_update_rate: int,
                 gradient_clip: Optional[float]=None, grad_report_rate: int=1000):
        self._online_network = model
        self._target_network = copy.deepcopy(self._online_network)
        self._target_update_rate = target_update_rate
        self._grad_report_rate = grad_report_rate
        self._gradient_clip = gradient_clip

        self._update_counter = 0
        self.name = ""

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._online_network.parameters()

    def update(self, batch: Batch_):
        loss = self._update(self.get_tensor_batch(batch))
        visualization.global_summary_writer.add_scalar(f'TD/TD loss ({self.name})', loss.data[0], self._update_counter)
        loss.backward()
        if self._update_counter % self._grad_report_rate == 0:
            for name, parameter in self._online_network.named_parameters():
                if parameter.grad is not None:
                    visualization.global_summary_writer.add_histogram(
                        f'{name} ({self.name})', parameter.data.cpu().numpy(), self._update_counter
                    )
                    visualization.global_summary_writer.add_histogram(
                        f'grad {name} ({self.name})', parameter.grad.data.cpu().numpy(), self._update_counter
                    )
        visualization.global_summary_writer.add_scalar(f'TD/TD loss ({self.name})', loss.data[0], self._update_counter)
        if self._gradient_clip is not None:
            for parameter in self._online_network.parameters():
                parameter.grad.data.clamp_(-self._gradient_clip, self._gradient_clip)
        # noinspection PyArgumentList
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
