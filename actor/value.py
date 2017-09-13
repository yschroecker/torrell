from typing import Any

import torch
import numpy as np

import policies.policy
import actor.actor_base
import visualization


class EpsilonGreedy(actor.actor_base.Actor, policies.policy.Policy[int]):
    def __init__(self, num_actions: int, q_model: torch.nn.Module,
                 initial_epsilon: float, final_epsilon: float, decay_rate: float):
        self.num_actions = num_actions

        self._final_epsilon = final_epsilon
        self._decay_delta = (initial_epsilon-final_epsilon)*decay_rate
        self._epsilon = initial_epsilon
        self._q_model = q_model

    @property
    def _model(self) -> torch.nn.Module:
        return self._q_model

    def update(self, _: Any):
        visualization.global_summary_writer.add_scalar(f'epsilon', self._epsilon)
        self._epsilon = max(self._epsilon - self._decay_delta, self._final_epsilon)

    def probabilities(self, states: torch.autograd.Variable, training: bool=True) -> np.ndarray:
        epsilon = self._epsilon if training else 0

        q_values = self._model(states)
        # noinspection PyArgumentList
        _, argmax = torch.max(q_values, dim=1)
        batch_size = states.size()[0]
        probabilities: torch.FloatTensor = torch.ones((batch_size, self.num_actions))*epsilon/self.num_actions
        arange = torch.arange(0, batch_size).type(torch.LongTensor)
        if self.cuda:
            probabilities = probabilities.cuda()
            arange = arange.cuda()
        probabilities[arange, argmax.data] += (1-epsilon)
        if self.cuda:
            return probabilities.cpu().numpy()[0]
        else:
            return probabilities.numpy()[0]

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable):
        NotImplementedError()

    def _sample(self, state_var: torch.autograd.Variable, training: bool=True) -> int:
        action_probabilities = self.probabilities(state_var, training)
        return np.random.choice(self.num_actions, p=action_probabilities)


def greedy(num_actions: int, q_model: torch.nn.Module) -> EpsilonGreedy:
    return EpsilonGreedy(num_actions, q_model, 0, 0, 0)
