from typing import Tuple, Sequence, Union
import abc

import torch

import data
import visualization


class OptimizationStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def iterate(self, iteration: int, batch: data.Batch[data.RLTransitionSequence]):
        pass


class GradientDescentStrategy(OptimizationStrategy, metaclass=abc.ABCMeta):
    def __init__(self, optimizer: torch.optim.Optimizer, gradient_clipping: float = None,
                 clip_norm: Union[int, bytes] = 'inf'):
        self._optimizer = optimizer
        self._gradient_clipping = gradient_clipping
        self._clip_norm = clip_norm

    @abc.abstractmethod
    def _targets(self, iteration: int, batch: data.Batch[data.RLTransitionSequence]) -> \
            Tuple[Sequence[torch.autograd.Variable], Sequence[Sequence[torch.nn.Parameter]]]:
        pass

    def iterate(self, iteration: int, batch: data.Batch[data.RLTransitionSequence]):
        self._optimizer.zero_grad()
        losses, parameters = self._targets(iteration, batch)
        for loss in losses:
            loss.backward()

        if self._gradient_clipping is not None:
            for parameter in parameters:
                torch.nn.utils.clip_grad_norm(parameter, self._gradient_clipping, self._clip_norm)

        visualization.global_summary_writer.add_scalar('LR', self._optimizer.param_groups[0]['lr'], iteration)

        self._optimizer.step()

