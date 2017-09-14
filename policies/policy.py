from typing import Generic, TypeVar
import abc

import torch
import numpy as np

import torch_util

ActionT = TypeVar('ActionT')


class Policy(Generic[ActionT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        pass

    @abc.abstractmethod
    def _sample(self, state: torch.autograd.Variable) -> ActionT:
        pass

    @property
    @abc.abstractmethod
    def _model(self) -> torch.nn.Module:
        pass

    @property
    def cuda(self) -> bool:
        return torch_util.module_is_cuda(self._model)

    def sample(self, state: np.ndarray, training: bool=False) -> ActionT:
        state_tensor = torch.from_numpy(np.array([state])).type(
            torch.cuda.FloatTensor if torch_util.module_is_cuda(self._model) else torch.FloatTensor)
        state_var = torch.autograd.Variable(state_tensor, volatile=True)
        return self._sample(state_var, training)

