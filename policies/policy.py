from typing import Generic, TypeVar
import abc

import torch
import numpy as np

ActionT = TypeVar('ActionT')


class Policy(Generic[ActionT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        pass

    @abc.abstractmethod
    def sample(self, state: np.ndarray) -> ActionT:
        pass

