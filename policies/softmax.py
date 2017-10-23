from typing import Sequence, Type

import torch
import torch.nn.functional
import numpy as np

import policies.policy


class SoftmaxPolicyModel(policies.policy.PolicyModel[int]):
    def __init__(self, network: torch.nn.Module):
        self._network = network

    @property
    def _module(self) -> torch.nn.Module:
        return self._network

    def _logits(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return torch.nn.functional.log_softmax(self._module(states))

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        action_probabilities = self._logits(states)
        return action_probabilities.gather(dim=1, index=actions.unsqueeze(1))

    def entropy(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        # noinspection PyArgumentList
        return -torch.sum(self._logits(states) * torch.exp(self._logits(states)), dim=1)

    def all_probabilities(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return torch.nn.functional.softmax(self._module(states))

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._network.parameters()

    @property
    def action_type(self) -> Type[np.dtype]:
        return np.int32


class SoftmaxPolicy(policies.policy.Policy):
    def __init__(self, policy_model: SoftmaxPolicyModel):
        self.__model = policy_model

    @property
    def _model(self) -> SoftmaxPolicyModel:
        return self.__model

    def _sample(self, state_var: torch.autograd.Variable, _: int, training: bool=False) -> int:
        probabilities_var = self._model.all_probabilities(state_var)
        probabilities = probabilities_var.data
        if self._model.is_cuda:
            probabilities = probabilities.cpu()
        probabilities = probabilities[0].numpy()
        return np.random.choice(probabilities.shape[0], p=probabilities)

