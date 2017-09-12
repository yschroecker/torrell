from typing import Any

import torch
import numpy as np

import policies.policy
import actor.actor_base
import torch_util


class EpsilonGreedy(actor.actor_base.Actor, policies.policy.Policy[int]):
    def __init__(self, num_actions: int, q_model: torch.nn.Module,
                 initial_epsilon: float, final_epsilon: float, decay_rate: float):
        self.num_actions = num_actions

        self._final_epsilon = final_epsilon
        self._decay_delta = (initial_epsilon-final_epsilon)*decay_rate
        self._epsilon = initial_epsilon
        self._q_model = q_model

    def update(self, _: Any):
        torch_util.global_summary_writer.add_scalar(f'epsilon', self._epsilon)
        self._epsilon = max(self._epsilon - self._decay_delta, self._final_epsilon)

    def probabilities(self, states: torch.autograd.Variable, training: bool=True) -> torch_util.Tensor:
        epsilon = self._epsilon if training else 0

        q_values = self._q_model(states)
        # noinspection PyArgumentList
        _, argmax = torch.max(q_values, dim=1)
        batch_size = states.size()[0]
        probabilities: torch.FloatTensor = torch.ones((batch_size, self.num_actions))*epsilon/self.num_actions
        probabilities = probabilities.type(torch_util.Tensor)
        probabilities[torch.arange(0, batch_size).type(torch.LongTensor), argmax.data] += (1-epsilon)
        return probabilities

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable):
        NotImplementedError()

    def sample(self, state: np.ndarray) -> int:
        action_probabilities = self.probabilities(
            torch.autograd.Variable(torch.from_numpy(np.atleast_2d(state)).type(torch_util.Tensor), volatile=True)
        )
        return np.random.choice(self.num_actions, p=action_probabilities.numpy()[0])


def greedy(num_actions: int, q_model: torch.nn.Module) -> EpsilonGreedy:
    return EpsilonGreedy(num_actions, q_model, 0, 0, 0)
