from typing import Generic, TypeVar, Tuple, Any

import abc

import gym
import numpy as np

ActionT = TypeVar('ActionT')


class Environment(Generic[ActionT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, action: ActionT) -> Tuple[np.ndarray, float, bool, Any]:
        pass

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        pass

# noinspection PyUnresolvedReferences
Environment.register(gym.core.Env)
