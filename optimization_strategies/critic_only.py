from typing import Union

import torch

from optimization_strategies.optimization_strategy import OptimizationStrategy
from critic.temporal_difference import Batch, TemporalDifferenceBase
import visualization


class CriticOnly(OptimizationStrategy):
    def __init__(self, optimizer: torch.optim.Optimizer, critic: TemporalDifferenceBase,
                 gradient_clipping: float = None, clip_norm: Union[int, bytes] = 'inf'):
        self._critic = critic
        self._optimizer = optimizer
        self._gradient_clipping = gradient_clipping
        self._clip_norm = clip_norm

    def iterate(self, iteration: int, batch: Batch):
        self._optimizer.zero_grad()
        loss = self._critic.update_loss(batch)
        loss.backward()
        # noinspection PyTypeChecker
        if self._gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm(self._critic.parameters, self._gradient_clipping, self._clip_norm)

        visualization.global_summary_writer.add_scalar('LR', self._optimizer.param_groups[0]['lr'], iteration)

        # noinspection PyTypeChecker
        self._optimizer.step()

