from typing import Sequence

import torch

from torrell import policies
from actor.actor_base import Actor, Batch


class LikelihoodRatioGradient(Actor):
    def __init__(self, policy: policies.policy.PolicyModel[int], entropy_regularization: float=0,
                 grad_report_rate: int=100):
        self._policy = policy
        self._entropy_regularization = entropy_regularization
        self._grad_report_rate = grad_report_rate

        self._update_counter = 0

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._policy.parameters

    def update_loss(self, batch: Batch):
        self._update_counter += 1
        advantages = torch.autograd.Variable(batch.advantages, requires_grad=False)
        states = torch.autograd.Variable(batch.states, requires_grad=False)
        actions = torch.autograd.Variable(batch.actions, requires_grad=False)

        loss = (-self._policy.log_probability(states, actions).squeeze() * advantages).mean()
        if self._entropy_regularization > 0:
            loss -= self._entropy_regularization * self._policy.entropy(states).mean()

        # print(loss)
        if self._update_counter % self._grad_report_rate == 0:
            self._policy.visualize(self._update_counter)
        return loss
