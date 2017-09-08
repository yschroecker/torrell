import abc
from typing import NamedTuple

import torch_util


class Batch(NamedTuple):
    states: torch_util.Tensor
    actions: torch_util.LongTensor
    advantages: torch_util.Tensor


class Actor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, batch: Batch):
        pass
