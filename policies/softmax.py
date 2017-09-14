import torch
import torch.nn.functional
import numpy as np

import policies.policy


class SoftmaxPolicy(policies.policy.Policy[int]):
    def __init__(self, network: torch.nn.Module):
        self._network = network

    @property
    def _model(self) -> torch.nn.Module:
        return self._network

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        action_probabilities = torch.nn.functional.log_softmax(self._model(states))
        return action_probabilities.gather(dim=1, index=actions.unsqueeze(1))

    def _sample(self, state_var: torch.autograd.Variable, training: bool=False) -> int:
        probabilities_var = torch.nn.functional.softmax(self._model(state_var))
        probabilities = probabilities_var.data
        if self.cuda:
            probabilities = probabilities.cpu()
        probabilities = probabilities[0].numpy()
        return np.random.choice(probabilities.shape[0], p=probabilities)
