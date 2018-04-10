from typing import Any, Union, Tuple, Sequence

import torch

from core_algorithms.optimization_strategy import GradientDescentStrategy
from critic.advantages import AdvantageProvider
from actor.actor_base import Actor

import data


class ActorOnly(GradientDescentStrategy):
    def __init__(self, optimizer: torch.optim.Optimizer, actor: Actor,
                 advantage_provider: AdvantageProvider, gradient_clipping: float = None,
                 clip_norm: Union[int, bytes] = 'inf'):
        super().__init__(optimizer, gradient_clipping, clip_norm)
        self._actor = actor
        self._advantage_provider = advantage_provider

    def _targets(self, _: Any, batch: data.Batch[data.RLTransitionSequence]) -> \
            Tuple[Sequence[torch.autograd.Variable], Sequence[Sequence[torch.nn.Parameter]]]:
        is_cuda = any(param.is_cuda for param in self._actor.parameters)
        batch = batch.to_tensor(is_cuda)
        advantage_batch = self._advantage_provider.compute_advantages(batch)
        actor_loss = self._actor.update_loss(advantage_batch)
        return [actor_loss], [self._actor.parameters]
