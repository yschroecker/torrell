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
        states = batch.states()
        states_var = torch.autograd.Variable(states, volatile=True)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states(), volatile=True)
        intermediate_rewards = torch.autograd.Variable(batch.intermediate_returns(), volatile=True)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights(), volatile=True)

        advantages = intermediate_rewards + bootstrap_weights * self._td.values(bootstrap_states) - \
            self._td.values(states_var)
        return ActorBatch(states=states, actions=batch.actions(), advantages=advantages.data)


class OneStepQAdvantageProvider(AdvantageProvider):
    def __init__(self, td: critic.value_td.ValueTD, mean_reward_lr: float):
        self._td = td
        self._mean_reward_lr = mean_reward_lr
        self._mean_reward = None

    def compute_advantages(self, batch: data.Batch[data.TensorRLTransitionSequence]) -> ActorBatch:
        states = batch.states()
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states(), volatile=True)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights(), volatile=True)
        if self._mean_reward is None:
            self._mean_reward = batch.rewards().mean()
        else:
            self._mean_reward += self._mean_reward_lr * (batch.rewards().mean() - self._mean_reward)
        intermediate_rewards = data.sequence_returns([sequence.rewards - self._mean_reward
                                                      for sequence in batch.sequences], batch.discount_factor)
        intermediate_rewards = torch.autograd.Variable(intermediate_rewards, volatile=True)

        advantages = intermediate_rewards + bootstrap_weights * self._td.values(bootstrap_states)
        return ActorBatch(states=states, actions=batch.actions(), advantages=advantages.data)


class MeanRewardMonteCarlo(AdvantageProvider):
    def __init__(self, mean_reward_update_rate: float):
        self._mean_lr = mean_reward_update_rate

        self._mean_reward = 0

    def compute_advantages(self, batch: data.Batch[data.TensorRLTransitionSequence]) -> ActorBatch:
        self._mean_reward = self._mean_lr * torch.mean(batch.rewards()) + (1 - self._mean_lr) * self._mean_reward
        reward_sequences = [sequence.rewards - self._mean_reward for sequence in batch.sequences]
        returns = data.sequence_returns(reward_sequences, batch.discount_factor)

        return ActorBatch(states=batch.states(), actions=batch.actions(), advantages=returns)
