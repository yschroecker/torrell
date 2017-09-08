import torch
import torch.nn.functional
import numpy as np

import policies.policy
import torch_util


class SoftmaxPolicy(policies.policy.Policy[int]):
    def __init__(self, network: torch.nn.Module):
        self._network = network

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        action_probabilities = torch.nn.functional.log_softmax(self._network(states))
        return action_probabilities.gather(dim=1, index=actions.unsqueeze(1))

    def sample(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(np.atleast_2d(state)).type(torch_util.Tensor)
        state_var = torch.autograd.Variable(state_tensor, volatile=True)
        probabilities_var = torch.nn.functional.softmax(self._network(state_var))
        probabilities = probabilities_var.data[0].numpy()
        return np.random.choice(probabilities.shape[0], p=probabilities)
