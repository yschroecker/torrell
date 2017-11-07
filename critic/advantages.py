from typing import Optional
import abc

import critic.value_td
from actor.actor_base import Batch as ActorBatch

import torch
import data


class AdvantageProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_advantages(self, batch: data.Batch[data.TensorRLTransitionSequence]) -> Optional[ActorBatch]:
        pass


class TDErrorAdvantageProvider(AdvantageProvider):
    def __init__(self, td: critic.value_td.ValueTD):
        self._td = td

    def compute_advantages(self, batch: data.Batch[data.TensorRLTransitionSequence]) -> ActorBatch:
        states = torch.autograd.Variable(batch.states(), volatile=True)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states(), volatile=True)
        intermediate_rewards = torch.autograd.Variable(batch.intermediate_returns(), volatile=True)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights(), volatile=True)

        advantages = intermediate_rewards + bootstrap_weights * self._td.values(bootstrap_states) - \
            self._td.values(states)
        return ActorBatch(states=batch.states(), actions=batch.actions(), advantages=advantages.data)
