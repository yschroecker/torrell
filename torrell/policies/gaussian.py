from typing import Sequence, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional
import torch_util

from torrell import policies


class SphericalGaussianPolicyModel(policies.policy.PolicyModel[np.ndarray]):
    def __init__(self, action_dim: int, network: torch.nn.Module, fixed_noise: Optional[np.ndarray],
                 min_stddev: float=0, noise_sample_rate: int=1, eval_use_mean: bool=False):
        self._network = network
        if fixed_noise is None:
            self._fixed_noise = None
        elif type(fixed_noise) is float or type(fixed_noise) is int:
            self._fixed_noise = torch.autograd.Variable(torch.ones(action_dim) * fixed_noise)
            if self.is_cuda:
                self._fixed_noise = self._fixed_noise.cuda()
        else:
            self._fixed_noise = torch.autograd.Variable(torch_util.load_input(self.is_cuda, fixed_noise))
        self._action_dim = action_dim
        self.min_stddev = min_stddev
        self._noise_sample_rate = noise_sample_rate
        self._noise = None
        self.eval_use_mean = eval_use_mean

    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        means, logstddevs = self.statistics(states)
        lognorm_constants = -0.5 * np.asscalar(self._action_dim*np.log(2*np.pi).astype(np.float32)) - \
            logstddevs.sum(dim=1)
        log_pdf = -0.5 * (((actions - means)/torch.exp(logstddevs))**2).sum(dim=1) + lognorm_constants
        return log_pdf

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._network.parameters()

    def entropy(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        mean, logstddevs = self.statistics(states)
        return np.asscalar(0.5 + np.log(np.sqrt(2*np.pi)).astype(np.float32))*self._action_dim + logstddevs.sum(dim=1)

    @property
    def _module(self) -> torch.nn.Module:
        return self._network

    def statistics(self, states: torch.autograd.Variable) -> Tuple[torch.autograd.Variable, torch.autograd.Variable]:
        out = self._network(states)
        if self._fixed_noise is not None:
            means = out
            logstddevs = self._fixed_noise.expand_as(means)
        else:
            means = out[:, :self._action_dim]
            logstddevs = out[:, self._action_dim:]
        return means, logstddevs

    @property
    def action_type(self) -> Type[np.dtype]:
        return np.float32


class SphericalGaussianPolicy(policies.policy.Policy[np.ndarray]):
    def __init__(self, model: SphericalGaussianPolicyModel):
        self.__model = model

    @property
    def _model(self) -> SphericalGaussianPolicyModel:
        return self.__model

    def sample_from_var(self, state: torch.autograd.Variable, t: int, training: bool):
        mean_var, logstddev_var = self._model.statistics(state)
        mean_tensor = mean_var.data
        logstddev_tensor = logstddev_var.data
        if self._model.is_cuda:
            mean_tensor = mean_tensor.cpu()
            logstddev_tensor = logstddev_tensor.cpu()
        mean = mean_tensor.numpy().squeeze()
        if not training and self.__model.eval_use_mean:
            action = mean
        else:
            logstddev = logstddev_tensor.numpy().squeeze()

            stddev = np.exp(logstddev) + self._model.min_stddev
            action = np.random.normal(mean, stddev)

        return action.astype(np.float32)

