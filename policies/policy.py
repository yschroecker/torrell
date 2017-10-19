from typing import Generic, TypeVar, Sequence
import abc

import torch
import numpy as np

import torch_util
import visualization

ActionT = TypeVar('ActionT')


class Policy(Generic[ActionT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log_probability(self, states: torch.autograd.Variable, actions: torch.autograd.Variable) -> \
            torch.autograd.Variable:
        pass

    @abc.abstractmethod
    def entropy(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        pass

    @abc.abstractmethod
    def _sample(self, state: torch.autograd.Variable, t: int, training: bool) -> ActionT:
        pass

    @property
    @abc.abstractmethod
    def _model(self) -> torch.nn.Module:
        pass

    @property
    def cuda(self) -> bool:
        return torch_util.module_is_cuda(self._model)

    @property
    @abc.abstractmethod
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        pass

    def visualize(self, counter):
        for name, parameter in self._model.named_parameters():
            if parameter.grad is not None:
                visualization.global_summary_writer.add_histogram(
                    f'pi ({name})', parameter.data.cpu().numpy(), counter
                )
                visualization.global_summary_writer.add_histogram(
                    f'pi {name}_grad', parameter.grad.data.cpu().numpy(), counter
                )

    def sample(self, state: np.ndarray, t: int, training: bool=False) -> ActionT:
        state_tensor = torch.from_numpy(np.array([state])).type(
            torch.cuda.FloatTensor if torch_util.module_is_cuda(self._model) else torch.FloatTensor)
        state_var = torch.autograd.Variable(state_tensor, volatile=True)
        return self._sample(state_var, t, training)

