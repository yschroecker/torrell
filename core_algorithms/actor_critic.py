from typing import Any, Union, Tuple, Sequence

import torch

from core_algorithms.optimization_strategy import GradientDescentStrategy
from critic.temporal_difference import Batch, TemporalDifferenceBase
from critic.advantages import AdvantageProvider
from actor.actor_base import Actor

import data


class ActorCritic(GradientDescentStrategy):
    def __init__(self, optimizer: torch.optim.Optimizer, actor: Actor, critic: TemporalDifferenceBase,
                 advantage_provider: AdvantageProvider, gradient_clipping: float = None,
                 clip_norm: Union[int, bytes] = 'inf'):
        super().__init__(optimizer, gradient_clipping, clip_norm)
        self._actor = actor
        self._critic = critic
        self._advantage_provider = advantage_provider

    def _targets(self, _: Any, batch: data.Batch[data.RLTransitionSequence]) -> \
            Tuple[Sequence[torch.autograd.Variable], Sequence[Sequence[torch.nn.Parameter]]]:
        # batch = data.new_to_old_tensor(batch.to_tensor(True))
        batch = self._critic.get_tensor_batch(batch)
        critic_loss = self._critic.update_loss(batch)
        advantage_batch = self._advantage_provider.compute_advantages(batch)
        actor_loss = self._actor.update_loss(advantage_batch)
        return [critic_loss, actor_loss], [self._critic.parameters, self._actor.parameters]
