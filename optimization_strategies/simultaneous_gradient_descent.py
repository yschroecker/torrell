from typing import Union

import torch

from optimization_strategies.optimization_strategy import OptimizationStrategy
from critic.temporal_difference import Batch, TemporalDifferenceBase
from critic.advantages import AdvantageProvider
from actor.actor_base import Actor
import visualization


class SimultaneousGradientDescent(OptimizationStrategy):
    def __init__(self, optimizer: torch.optim.Optimizer, actor: Actor, critic: TemporalDifferenceBase,
                 advantage_provider: AdvantageProvider, gradient_clipping: float = None,
                 clip_norm: Union[int, bytes] = 'inf'):
        self._actor = actor
        self._critic = critic
        self._advantage_provider = advantage_provider
        self._optimizer = optimizer
        self._gradient_clipping = gradient_clipping
        self._clip_norm = clip_norm

    def iterate(self, iteration: int, batch: Batch):
        self._optimizer.zero_grad()
        critic_loss = self._critic.update_loss(batch)
        critic_loss.backward()
        advantage_batch = self._advantage_provider.compute_advantages(self._critic.get_tensor_batch(batch))
        actor_loss = self._actor.update_loss(advantage_batch)
        actor_loss.backward()

        if self._gradient_clipping is not None:
            # noinspection PyTypeChecker
            torch.nn.utils.clip_grad_norm(self._critic.parameters, self._gradient_clipping, self._clip_norm)
            torch.nn.utils.clip_grad_norm(self._actor.parameters, self._gradient_clipping, self._clip_norm)

        visualization.global_summary_writer.add_scalar('LR', self._optimizer.param_groups[0]['lr'], iteration)

        # noinspection PyTypeChecker
        self._optimizer.step()
