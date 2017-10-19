from typing import Sequence, Optional

import torch
import torch.nn.functional
import numpy as np

import policies.policy


class SphericalGaussianPolicy(policies.policy.Policy[np.ndarray]):
    def __init__(self, action_dim: int, network: torch.nn.Module, fixed_noise: Optional[np.ndarray],
                 min_stddev: float=0):
        self._network = network
        self._fixed_noise = fixed_noise
        self._action_dim = action_dim
        self._min_stddev = min_stddev

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        out = self._network(states)
        means = out[:, :self._action_dim]
        logstddevs = out[:, self._action_dim:]
        lognorm_constants = -0.5 * np.asscalar(self._action_dim*np.log(2*np.pi).astype(np.float32)) - \
            logstddevs.sum(dim=1)
        log_pdf = -0.5 * (((actions - means)/torch.exp(logstddevs))**2).sum(dim=1) + lognorm_constants
        return log_pdf

    def _sample(self, state: torch.autograd.Variable, _) -> np.ndarray:
        out = self._network(state).data
        if self.cuda:
            out = out.cpu()
        out = out.numpy()

        logstddev = self._fixed_noise
        if logstddev is None:
            logstddev = out[0, self._action_dim:]
        stddev = np.exp(logstddev) + self._min_stddev
        mean = out[0, :self._action_dim]
        action = np.random.normal(mean, stddev)
        return action.astype(np.float32)

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._network.parameters()

    def entropy(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        NotImplementedError()

    @property
    def _model(self) -> torch.nn.Module:
        return self._network
