import abc

import torch


class DiscreteActor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def probabilities(self, states: torch.autograd.Variable, training: bool=True) -> torch.autograd.Variable:
        pass
