from typing import Tuple, Any, Union

import numpy as np


class Tabular:
    def __init__(self, transition_matrix: np.ndarray, reward_matrix: np.ndarray, terminal_states: np.ndarray,
                 initial_states: np.ndarray):
        self.num_states = transition_matrix.shape[1]
        self.num_actions = transition_matrix.shape[0]//self.num_states
        self._transition_matrix = transition_matrix
        self._reward_matrix = reward_matrix
        self._terminal_states = terminal_states
        self._initial_states = initial_states

    def sample_initial(self) -> int:
        return np.random.choice(self.num_states, p=self._initial_states)

    def transition(self, state: int, action: int) -> int:
        return np.random.choice(self.num_states, p=self._transition_matrix[self._sa_index(state, action)])

    def reward(self, state: int, action: int, next_state: int) -> float:
        return self._reward_matrix[self._sa_index(state, action), next_state]

    def is_terminal(self, state: int) -> bool:
        return self._terminal_states[state]

    def _policy_transition_matrix(self, policy: np.ndarray) -> np.ndarray:
        policy_selector = np.zeros((self.num_states, self.num_states * self.num_actions))
        for state in range(self.num_states):
            policy_selector[state, self._sa_index(state, policy[state])] = 1
        return np.dot(policy_selector, self._transition_matrix)

    def _sa_index(self, state: int, action: int) -> int:
        return self.num_actions * state + action


def state_reward_to_matrix(reward_vector: np.ndarray, num_actions: int) -> np.ndarray:
    return np.repeat(np.atleast_2d(reward_vector), len(reward_vector) * num_actions, axis=0)


class TabularEnv:
    def __init__(self, tabular: Tabular):
        self._tabular = tabular

        self._state = self.reset()

    def reset(self) -> int:
        self._state = self._tabular.sample_initial()
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, Any]:
        next_state = self._tabular.transition(self._state, action)
        reward = self._tabular.reward(self._state, action, next_state)
        self._state = next_state
        return self._state, reward, self._tabular.is_terminal(self._state), None

    @property
    def num_states(self) -> int:
        return self._tabular.num_states

    @property
    def num_actions(self) -> int:
        return self._tabular.num_actions


class OneHotEnv:
    def __init__(self, env: TabularEnv):
        self._env = env
        self._eye = np.eye(self._env.num_states)

    @classmethod
    def from_tabular(cls, tabular: Tabular):
        return cls(TabularEnv(tabular))

    def reset(self) -> np.ndarray:
        return self._eye(self._env.reset())

    def step(self, action) -> Tuple[np.ndarray, float, bool, Any]:
        state, reward, is_terminal, info = self._env.step(action)
        return self._eye[state], reward, is_terminal, info


G = 1
S = 2
W = 3


class Gridworld:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def _state_index(self, x: int, y: int) -> int:
        return self.width * y + x

    def _pos(self, state_index: int) -> Tuple[int, int]:
        return state_index % self.width, state_index // self.width

    def _consolidate_pos(self, transition_grid, new, original):
        if self.width > new[0] >= 0 and self.height > new[1] >= 0 and transition_grid[new] != W:
            return new
        else:
            return original

    def __init__(self, transition_grid: np.ndarray, reward_grid: np.ndarray, transition_noise: float=0):
        self.height = transition_grid.shape[0]
        self.width = transition_grid.shape[1]
        self.num_states = self.width * self.height
        self.num_actions = 4

        # Transition matrix
        transition_matrix = np.zeros((self.num_states * self.num_actions, self.num_states))
        for state in range(self.num_states):
            pos = self._pos(state)
            left = self._state_index(*self._consolidate_pos(transition_grid, (pos[0] - 1, pos[1]), pos))
            right = self._state_index(*self._consolidate_pos(transition_grid, (pos[0] + 1, pos[1]), pos))
            up = self._state_index(*self._consolidate_pos(transition_grid, (pos[0], pos[1] - 1), pos))
            down = self._state_index(*self._consolidate_pos(transition_grid, (pos[0], pos[1] + 1), pos))

            transition_matrix[state * self.num_actions + self.LEFT, left] += 1 - transition_noise
            transition_matrix[state * self.num_actions + self.LEFT, up] += transition_noise/2
            transition_matrix[state * self.num_actions + self.LEFT, down] += transition_noise/2

            transition_matrix[state * self.num_actions + self.RIGHT, right] += 1 - transition_noise
            transition_matrix[state * self.num_actions + self.RIGHT, up] += transition_noise/2
            transition_matrix[state * self.num_actions + self.RIGHT, down] += transition_noise/2

            transition_matrix[state * self.num_actions + self.UP, up] += 1 - transition_noise
            transition_matrix[state * self.num_actions + self.UP, left] += transition_noise/2
            transition_matrix[state * self.num_actions + self.UP, right] += transition_noise/2

            transition_matrix[state * self.num_actions + self.DOWN, up] += 1 - transition_noise
            transition_matrix[state * self.num_actions + self.DOWN, left] += transition_noise/2
            transition_matrix[state * self.num_actions + self.DOWN, right] += transition_noise/2

        # Reward matrix
        reward_vector = np.zeros((self.num_states,))
        terminal_vector = np.zeros((self.num_states,))
        initial_vector = np.zeros((self.num_states,))
        for state in range(self.num_states):
            reward_vector[state] = reward_grid[self._pos(state)]
            terminal_vector[state] = transition_grid[self._pos(state)] == G
            initial_vector[state] = transition_grid[self._pos(state)] == S
        reward_matrix = state_reward_to_matrix(reward_vector, self.num_actions)
        initial_vector /= np.sum(initial_vector)

        self._tabular = Tabular(transition_matrix, reward_matrix, terminal_vector, initial_vector)

    def tabular_env(self):
        return TabularEnv(self._tabular)

    def one_hot_env(self):
        return OneHotEnv.from_tabular(self._tabular)

    def state_repr(self, state: Union[int, np.ndarray]) -> Tuple[int, int]:
        if type(state) is int:
            return self._pos(state)
        else:
            return self._pos(np.asscalar(np.argmax(state)))


simple_grid1 = Gridworld(
    np.array(
        [[0, 0, 0, 0, G],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [S, 0, 0, 0, 0]]
    ).T,
    np.array(
        [[0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    ).T
)


if __name__ == '__main__':
    env = simple_grid1.tabular_env()
    state = env.reset()
    print(type(state))
    print(simple_grid1.state_repr(state))
    state, reward, is_terminal, _ = env.step(Gridworld.RIGHT)
    print(simple_grid1.state_repr(state))

