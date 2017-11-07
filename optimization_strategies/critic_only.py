from typing import Union, Tuple, Sequence, Any

import torch

from optimization_strategies.optimization_strategy import GradientDescentStrategy
from critic.temporal_difference import TemporalDifferenceBase
import data


class CriticOnly(GradientDescentStrategy):
    def __init__(self, optimizer: torch.optim.Optimizer, critic: TemporalDifferenceBase,
                 gradient_clipping: float = None, clip_norm: Union[int, bytes] = 'inf'):
        super().__init__(optimizer, gradient_clipping, clip_norm)
        self._critic = critic
        self._optimizer = optimizer
        self._gradient_clipping = gradient_clipping
        self._clip_norm = clip_norm

    def _targets(self, _: Any, batch: data.Batch[data.RLTransitionSequence]) -> \
            Tuple[Sequence[torch.autograd.Variable], Sequence[Sequence[torch.nn.Parameter]]]:
        critic_loss = self._critic.update_loss(batch)
        return [critic_loss], [self._critic.parameters]
