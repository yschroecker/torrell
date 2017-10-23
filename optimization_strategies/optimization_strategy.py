import abc

from critic.temporal_difference import Batch


class OptimizationStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def iterate(self, iteration: int, batch: Batch):
        pass
