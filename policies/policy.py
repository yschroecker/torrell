from typing import Generic, TypeVar, Sequence, Type
import abc

import torch
import numpy as np

import torch_util
import visualization

ActionT = TypeVar('ActionT')


class PolicyModel(Generic[ActionT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        pass

    @abc.abstractmethod
    def entropy(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        pass

    @property
    @abc.abstractmethod
    def _module(self) -> torch.nn.Module:
        pass

    @property
    def is_cuda(self) -> bool:
        return torch_util.module_is_cuda(self._module)

    @property
    @abc.abstractmethod
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        pass

    def visualize(self, counter):
        for name, parameter in self._module.named_parameters():
            visualization.global_summary_writer.add_histogram(
                f'pi ({name})', parameter.data.cpu().numpy(), counter
            )
            if parameter.grad is not None:
                visualization.global_summary_writer.add_histogram(
                    f'pi {name}_grad', parameter.grad.data.cpu().numpy(), counter
                )

    @property
    @abc.abstractmethod
    def action_type(self) -> Type[np.dtype]:
        pass


class Policy(Generic[ActionT], metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def _model(self) -> PolicyModel:
        pass

    @abc.abstractmethod
    def sample_from_var(self, state: torch.autograd.Variable, t: int, training: bool) -> ActionT:
        pass

    def sample(self, state: np.ndarray, t: int, training: bool=True) -> ActionT:
        state_tensor = torch.from_numpy(np.array([state])).type(
            torch.cuda.FloatTensor if self._model.is_cuda else torch.FloatTensor
        )
        state_var = torch.autograd.Variable(state_tensor, volatile=True)
        return self.sample_from_var(state_var, t, training)

    @property
    def action_type(self) -> Type[np.dtype]:
        return self._model.action_type

