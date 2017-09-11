import abc
from typing import NamedTuple

import torch_util


class Batch(NamedTuple):
    states: torch_util.FloatTensor
    actions: torch_util.LongTensor
    advantages: torch_util.FloatTensor


class Actor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, batch: Batch):
        pass
