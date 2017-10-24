from typing import Any, Sequence, Callable, Type

import torch
import numpy as np

import policies.policy
import actor.actor_base
import visualization


class ValuePolicyModel(policies.policy.PolicyModel[int]):
    def __init__(self, num_actions: int, q_model: torch.nn.Module):
        self.num_actions = num_actions

        self._q_model = q_model

    @property
    def _module(self) -> torch.nn.Module:
        return self._q_model

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable):
        NotImplementedError()

    def entropy(self, states: torch.autograd.Variable):
        NotImplementedError()

    def q_values(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return self._module(states)

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return []

    @property
    def action_type(self) -> Type[np.dtype]:
        return np.int32


class EpsilonGreedy(policies.policy.Policy[int]):
    def __init__(self, value_model: ValuePolicyModel, initial_epsilon: float, final_epsilon: float, decay_rate: float):
        self.__model = value_model
        self._final_epsilon = final_epsilon
        self._decay_delta = (initial_epsilon - final_epsilon) * decay_rate
        self._epsilon = initial_epsilon

    @property
    def _model(self):
        return self.__model

    def probabilities(self, states: torch.autograd.Variable, training: bool = True) -> np.ndarray:
        epsilon = self._epsilon if training else 0

        q_values = self._model.q_values(states)
        # noinspection PyArgumentList
        _, argmax = torch.max(q_values, dim=1)
        batch_size = states.size()[0]
        probabilities: torch.FloatTensor = torch.ones((batch_size, self._model.num_actions)) * \
            epsilon / self._model.num_actions
        arange = torch.arange(0, batch_size).type(torch.LongTensor)
        if self._model.is_cuda:
            probabilities = probabilities.cuda()
            arange = arange.cuda()
        probabilities[arange, argmax.data] += (1 - epsilon)
        if self._model.is_cuda:
            return probabilities.cpu().numpy()[0]
        else:
            return probabilities.numpy()[0]

    def _sample(self, state_var: torch.autograd.Variable, _: Any, training: bool=True) -> int:
        if training:
            visualization.global_summary_writer.add_scalar(f'epsilon', self._epsilon)
            self._epsilon = max(self._epsilon - self._decay_delta, self._final_epsilon)

        action_probabilities = self.probabilities(state_var, training)
        return np.random.choice(self._model.num_actions, p=action_probabilities)


def epsilon_greedy(initial_epsilon: float, final_epsilon: float, decay_rate: float) -> \
        Callable[[ValuePolicyModel], EpsilonGreedy]:
    def _build(value_model: ValuePolicyModel) -> EpsilonGreedy:
        return EpsilonGreedy(value_model, initial_epsilon, final_epsilon, decay_rate)
    return _build


def greedy() -> Callable[[ValuePolicyModel], EpsilonGreedy]:
    return epsilon_greedy(0, 0, 0)
