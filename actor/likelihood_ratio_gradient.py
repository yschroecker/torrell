import torch

import policies.policy
from actor.actor_base import Batch


class LikelihoodRatioGradient:
    requires_advantages = True

    def __init__(self, policy: policies.policy.Policy[int], optimizer: torch.optim.Optimizer):
        self._policy = policy
        self._optimizer = optimizer

    def update(self, batch: Batch):
        advantages = torch.autograd.Variable(batch.advantages, requires_grad=False)
        states = torch.autograd.Variable(batch.states, requires_grad=False)
        actions = torch.autograd.Variable(batch.actions, requires_grad=False)

        loss = (-self._policy.log_probability(states, actions).squeeze() * advantages).mean()

        self._optimizer.zero_grad()
        loss.backward()
        # noinspection PyArgumentList
        self._optimizer.step()
