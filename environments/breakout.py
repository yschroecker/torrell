from typing import Tuple, Any, Optional
import collections

import numpy as np
import gym
import skimage.color
import skimage.transform

import environments.environment


class Breakout(environments.environment.Environment[int]):
    history_length = 4
    image_height = 67
    image_width = 47
    state_dim = [4, 67, 47]

    def __init__(self, monitor_out: Optional[str]=None):
        self._env = gym.make('Breakout-v0')
        if monitor_out is not None:
            self._env = gym.wrappers.Monitor(self._env, monitor_out, force=True)
        self.num_actions = self._env.action_space.n
        self._history = collections.deque(maxlen=self.history_length)

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        image = image[5:-5, 10:-10]
        image = skimage.color.rgb2gray(image)
        image = skimage.transform.downscale_local_mean(image, (3, 3))
        return image

    def reset(self) -> np.ndarray:
        self._env.reset()
        # for _ in range(0):
            # self._env.step(np.random.choice(self.num_actions))
        assert self.history_length > 0
        for _ in range(self.history_length):
            state = self.step(np.random.choice(self.num_actions))[0]
        # noinspection PyUnboundLocalVariable
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Any]:
        state, reward, is_terminal, info = self._env.step(action)
        state = self._preprocess(state)
        self._history.append(state)
        state = np.array(self._history)
        return state, reward, is_terminal, info


def _test_preprocessing():
    env = Breakout()
    state = env.reset()
    for _ in range(100):
        state, _, _, _ = env.step(np.random.choice(env.num_actions))

    import scipy.misc
    scipy.misc.imshow(state[0])


if __name__ == '__main__':
    _test_preprocessing()
