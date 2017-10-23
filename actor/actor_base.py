import abc
from typing import NamedTuple, Sequence

import torch

import torch_util


class Batch(NamedTuple):
    states: torch_util.FloatTensor
    actions: torch_util.LongTensor
    advantages: torch_util.FloatTensor


class Actor(metaclass=abc.ABCMeta):
    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        pass

    @abc.abstractmethod
    def update_loss(self, batch: Batch):
        pass
