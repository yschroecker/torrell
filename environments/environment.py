from typing import Generic, TypeVar, Tuple, Any

import abc

import gym
import chainerrl.envs.ale
import numpy as np

ActionT = TypeVar('ActionT')


class Environment(Generic[ActionT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, action: ActionT) -> Tuple[np.ndarray, float, bool, Any]:
        pass

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        pass


class RewardClippedEnvironment(Environment[ActionT]):
    def __init__(self, env: Environment[ActionT],
                 reward_clipping: Tuple[float, float] = (-np.float('inf'), np.float('inf'))):
        self._reward_clipping = reward_clipping
        self._env = env

    def step(self, action: ActionT) -> Tuple[np.ndarray, float, bool, Any]:
        next_state, reward, is_terminal, info = self._env.step(action)
        reward = np.clip(reward, self._reward_clipping[0], self._reward_clipping[1])
        return next_state, reward, is_terminal, info

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        return self._env.reset()


# noinspection PyUnresolvedReferences
Environment.register(gym.core.Env)
# noinspection PyUnresolvedReferences
Environment.register(chainerrl.envs.ale.ALE)
