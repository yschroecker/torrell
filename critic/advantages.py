from typing import Optional
import abc

from critic.temporal_difference import TensorBatch as CriticBatch
import critic.value_td
from actor.actor_base import Batch as ActorBatch

import torch


class AdvantageProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_advantages(self, batch: CriticBatch) -> Optional[ActorBatch]:
        pass


class NoAdvantageProvider(AdvantageProvider):
    def compute_advantages(self, batch: CriticBatch):
        return None


class TDErrorAdvantageProvider(AdvantageProvider):
    def __init__(self, td: critic.value_td.ValueTD):
        self._td = td

    def compute_advantages(self, batch: CriticBatch) -> ActorBatch:
        states = torch.autograd.Variable(batch.states, volatile=True)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states, volatile=True)
        intermediate_rewards = torch.autograd.Variable(batch.intermediate_returns, volatile=True)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights, volatile=True)
        is_terminal = (bootstrap_weights > 1e-5).type(torch.FloatTensor)  # TODO: refactor!!!

        advantages = (intermediate_rewards + bootstrap_weights * self._td.values(bootstrap_states) - \
                      self._td.values(states)) * is_terminal
        return ActorBatch(states=batch.states, actions=batch.actions, advantages=advantages.data)
