from typing import Optional, Sequence, Any
import abc
import copy

import numpy as np
import torch

import data
import torch_util
import visualization


class TemporalDifferenceBase(metaclass=abc.ABCMeta):
    def __init__(self, model: torch.nn.Module, target_update_rate: int, grad_report_rate: int = 1000):
        self._online_network = model
        self._target_network = copy.deepcopy(self._online_network)
        self._target_update_rate = target_update_rate
        self._grad_report_rate = grad_report_rate

        self._update_counter = 0
        self.name = ""

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._online_network.parameters()

    def update_loss(self, batch: data.Batch[data.RLTransitionSequence]):
        loss = self._update(self.get_tensor_batch(batch))
        visualization.global_summary_writer.add_scalar(f'TD/TD loss ({self.name})', loss.data[0], self._update_counter)
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
        # noinspection PyArgumentList
        self._update_counter += 1
        if self._update_counter % self._target_update_rate == 0:
            self._target_network = copy.deepcopy(self._online_network)
        return loss

    def get_tensor_batch(self, batch: data.Batch[data.RLTransitionSequence]) -> \
            data.Batch[data.TensorRLTransitionSequence]:
        return batch.to_tensor(torch_util.module_is_cuda(self._online_network))

    @abc.abstractmethod
    def _update(self, batch: data.TensorRLTransitionSequence) -> torch.autograd.Variable:
        pass
