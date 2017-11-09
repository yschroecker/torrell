from typing import Tuple, Any, Union
import functools
import os
import tqdm

import numpy as np

import environments.environment


class Tabular:
    def __init__(self, transition_matrix: np.ndarray, reward_matrix: np.ndarray, terminal_states: np.ndarray,
                 initial_states: np.ndarray):
        self.num_states = transition_matrix.shape[1]
        self.num_actions = transition_matrix.shape[0] // self.num_states
        self.transition_matrix = transition_matrix
        self._reward_matrix = reward_matrix
        self.terminal_states = terminal_states.astype(bool)
        self._initial_states = initial_states

    def sample_initial(self) -> int:
        return np.random.choice(self.num_states, p=self._initial_states)

    def transition(self, state: int, action: int) -> int:
        return np.random.choice(self.num_states, p=self.transition_matrix[self._sa_index(state, action)])

    def reward(self, state: int, action: int, next_state: int) -> float:
        return self._reward_matrix[self._sa_index(state, action), next_state]

    def is_terminal(self, state: int) -> bool:
        return self.terminal_states[state]

    def _wrap_around_transition_matrix(self) -> np.ndarray:
        wrap_around_matrix = self.transition_matrix.copy()
        for state in range(self.num_states):
            if self.is_terminal(state):
                for action in range(self.num_actions):
                    sa = self._sa_index(state, action)
                    wrap_around_matrix[sa, :] = self._initial_states
        return wrap_around_matrix

    def deterministic_policy_transition_matrix(self, deterministic_policy: np.ndarray,
                                               transition_matrix: np.ndarray) -> np.ndarray:
        policy = np.zeros((self.num_states, self.num_states * self.num_actions))
        for state in range(self.num_states):
            policy[state, self._sa_index(state, deterministic_policy[state])] = 1
        return self.policy_transition_matrix(policy, transition_matrix)

    def reward_vector(self):
        rewards = np.zeros((self.num_states * self.num_actions,))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                rewards[state * self.num_actions + action] = \
                    self.transition_matrix[state * self.num_actions + action, :] @ \
                    self._reward_matrix[state * self.num_actions + action, :]
        return rewards

    @staticmethod
    def policy_transition_matrix(policy: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
        return np.dot(policy, transition_matrix)

    def stationary_state_distribution(self, policy: np.ndarray) -> np.ndarray:
        transition_matrix = self.policy_transition_matrix(policy, self._wrap_around_transition_matrix())
        values, vectors = np.linalg.eig(transition_matrix.T)
        for value, vector in zip(values, vectors.T):
            # noinspection PyTypeChecker
            if np.isclose(value, 1):
                if np.max(vector) <= 0:
                    vector *= -1
                if np.min(vector) < 0:
                    vector -= np.min(vector)
                return np.real(vector / np.sum(vector))
        assert False

    def reverse_transition_matrix(self, policy: np.ndarray, stationary_distribution: np.ndarray) -> np.ndarray:
        transition_matrix = self._wrap_around_transition_matrix()
        reverse_transitions = np.zeros((self.num_states, self.num_states * self.num_actions))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                for next_state in range(self.num_states):
                    sa = self._sa_index(state, action)
                    reverse_transitions[next_state, sa] = transition_matrix[sa, next_state] * policy[state, sa] * \
                        stationary_distribution[state] / stationary_distribution[next_state]
        return reverse_transitions

    def log_stationary_derivative(self, policy: np.ndarray, log_policy_derivative: np.ndarray,
                                  stationary_distribution: np.ndarray) -> np.ndarray:
        """
        :param policy: S x SA
        :param log_policy_derivative: SA X parameter_dims
        :param stationary_distribution: S
        :return: S X parameter_dims
        """
        parameter_dims = log_policy_derivative.shape[1:]
        reverse_transitions = self.reverse_transition_matrix(policy, stationary_distribution)
        reverse_state_transitions = sum(
            [reverse_transitions[:, np.arange(0, self.num_actions * self.num_states, self.num_actions) + i]
             for i in range(self.num_actions)]
        )
        # should be solve but is sometimes singular. Probably for numerical reasons?
        lsd = np.linalg.lstsq(np.eye(self.num_states) - reverse_state_transitions,
                              reverse_transitions @ np.reshape(log_policy_derivative,
                                                               newshape=[log_policy_derivative.shape[0], -1]))[0]
        lsd -= np.sum(lsd * stationary_distribution[:, np.newaxis], axis=0, keepdims=True)
        return np.reshape(lsd, newshape=(lsd.shape[0],) + parameter_dims)

    def _sa_index(self, state: int, action: int) -> int:
        return self.num_actions * state + action  # don't change


def policy_iteration(tab: Tabular, discount_factor: float) -> np.ndarray:
    policy = np.zeros((tab.num_states,), dtype=np.int32)
    rewards = tab.reward_vector()
    new_policy = np.argmax(rewards.reshape((-1, tab.num_actions)), axis=1)
    progress = tqdm.tqdm()
    while np.any(policy != new_policy):
        policy = new_policy.copy()
        transition_matrix = tab.transition_matrix.reshape(tab.num_states, tab.num_actions,
                                                          tab.num_states)[np.arange(tab.num_states), policy, :]
        policy_rewards = rewards.reshape(-1, tab.num_actions)[np.arange(tab.num_states), policy]
        long_term_probabilities: np.ndarray = np.eye(tab.num_states) - discount_factor * transition_matrix
        values = np.linalg.solve(long_term_probabilities, policy_rewards)
        # noinspection PyUnresolvedReferences
        q_values = rewards + discount_factor * tab.transition_matrix @ values
        new_policy = np.argmax(q_values.reshape((-1, tab.num_actions)), axis=1)
        progress.update()
    return policy


def value_iteration(tab: Tabular, discount_factor: float, eps: float = 1e-8) -> np.ndarray:
    rewards = tab.reward_vector()
    q_values = rewards.copy()
    values = np.max(q_values.reshape((-1, tab.num_actions)), axis=1)
    new_q_values = np.zeros((tab.num_states * tab.num_actions,))
    diff = eps + 1
    trange = tqdm.tqdm()
    while diff > eps:
        q_values = new_q_values
        np.argmax(q_values.reshape((-1, tab.num_actions)), axis=1)
        new_q_values = rewards + discount_factor * (1 - tab.terminal_states) * tab.transition_matrix @ values
        values = np.max(new_q_values.reshape((-1, tab.num_actions)), axis=1)
        diff = np.max((q_values - new_q_values) ** 2)
        trange.update()
        trange.set_description(f"diff: {diff}")
    return np.argmax(q_values.reshape((-1, tab.num_actions)), axis=1)


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


class OneHotEnv(environments.environment.Environment[int]):
    def __init__(self, env: TabularEnv):
        self._env = env
        self._eye = np.eye(self._env.num_states)

    @classmethod
    def from_tabular(cls, tabular: Tabular):
        return cls(TabularEnv(tabular))

    def reset(self) -> np.ndarray:
        return self._eye[self._env.reset()]

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

    def __init__(self, transition_grid: np.ndarray, reward_grid: np.ndarray, transition_noise: float = 0):
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
            transition_matrix[state * self.num_actions + self.LEFT, up] += transition_noise / 2
            transition_matrix[state * self.num_actions + self.LEFT, down] += transition_noise / 2

            transition_matrix[state * self.num_actions + self.RIGHT, right] += 1 - transition_noise
            transition_matrix[state * self.num_actions + self.RIGHT, up] += transition_noise / 2
            transition_matrix[state * self.num_actions + self.RIGHT, down] += transition_noise / 2

            transition_matrix[state * self.num_actions + self.UP, up] += 1 - transition_noise
            transition_matrix[state * self.num_actions + self.UP, left] += transition_noise / 2
            transition_matrix[state * self.num_actions + self.UP, right] += transition_noise / 2

            transition_matrix[state * self.num_actions + self.DOWN, down] += 1 - transition_noise
            transition_matrix[state * self.num_actions + self.DOWN, left] += transition_noise / 2
            transition_matrix[state * self.num_actions + self.DOWN, right] += transition_noise / 2

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

        self.tabular = Tabular(transition_matrix, reward_matrix, terminal_vector, initial_vector)

    def tabular_env(self):
        return TabularEnv(self.tabular)

    def one_hot_env(self):
        return OneHotEnv.from_tabular(self.tabular)

    def state_repr(self, state: Union[int, np.ndarray]) -> Tuple[int, int]:
        if type(state) is int:
            return self._pos(state)
        else:
            return self._pos(np.asscalar(np.argmax(state)))


class Racetrack:
    width = 33
    height = 9
    num_actions = 5

    @property
    def num_states(self):
        return self.width * self.height * 25 + 1

    def _transition_probabilities(self, state, action):
        assert 0 <= action <= 4

        next_x = state[0] + state[2]
        next_y = state[1] + state[3]
        next_x = min(self.width - 1, max(0, next_x))
        next_y = min(self.height - 1, max(0, next_y))

        if action == 4:
            success_probability = 1
        elif max(abs(state[2]), abs(state[3])) == 2:
            success_probability = 0.2
        else:
            success_probability = 0.9

        next_dx = state[2]
        next_dy = state[3]

        if action == 0:
            next_dx = state[2] + 1
        elif action == 1:
            next_dx = state[2] - 1
        elif action == 2:
            next_dy = state[3] + 1
        elif action == 3:
            next_dy = state[3] - 1

        next_dx = min(2, max(-2, next_dx))
        next_dy = min(2, max(-2, next_dy))

        possible_transitions = [(1 - success_probability, [next_x, next_y, state[2], state[3]]),
                                (success_probability, [next_x, next_y, next_dx, next_dy])]
        return possible_transitions

    def _state_from_index(self, state: int) -> Tuple[int, int, int, int]:
        dy = state % 5 - 2
        state //= 5
        dx = state % 5 - 2
        state //= 5
        x = state % self.width
        y = state // self.width
        return x, y, dx, dy

    def _state_index(self, state: Tuple[int, int, int, int]) -> int:
        x, y, dx, dy = state
        return dy + 2 + 5 * (dx + 2 + 5 * (x + self.width * y))

    @functools.lru_cache(None)
    def transition_probabilities(self, state, action):
        possible_transitions = self._transition_probabilities(self._state_from_index(state), action)
        probability_vector = np.zeros((self.num_states,))
        for p, next_state in possible_transitions:
            probability_vector[self._state_index(next_state)] += p
        return probability_vector

    def transition_matrix(self, reward_matrix: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.num_states * self.num_actions, self.num_states))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                matrix[state * self.num_actions + action, :] = self.transition_probabilities(state, action)
        indices: np.ndarray = reward_matrix[:-self.num_actions, :] == 5
        indices = np.argwhere(indices)
        for sa, next_state in indices:
            if next_state < self.num_states - 1:
                matrix[sa, -1] += matrix[sa, next_state]
                matrix[sa, next_state] = 0
        for action in range(self.num_actions):
            matrix[(self.num_states - 1) * self.num_actions + action, -1] = 1
        return matrix

    def _reward(self, state: Tuple[int, int, int, int], _: int, next_state: Tuple[int, int, int, int]):
        y = state[1]

        # Finish line
        reward = 0.
        if next_state[0] < 3:
            if y <= 5 and next_state[1] >= 6:
                reward = 5
            elif y >= 6 and next_state[1] <= 5:
                reward = -5

        if not (0 < next_state[0] < self.width - 2 and 0 < next_state[1] < self.height - 2):
            reward = min(reward, -0.1)

        if 3 <= next_state[0] < self.width - 3 and 3 <= next_state[1] < self.height - 3:
            reward = min(reward, -0.1)
        return reward

    def reward_matrix(self):
        matrix = np.zeros((self.num_states * self.num_actions, self.num_states))
        for state in range(self.num_states):
            for next_state in range(self.num_states):
                r = self._reward(self._state_from_index(state), 0, self._state_from_index(next_state))
                for action in range(self.num_actions):
                    matrix[state * self.num_actions + action, next_state] = r
        matrix[:-self.num_actions, -1] = 5
        matrix[-self.num_actions:, :] = 0

        return matrix

    def __init__(self):
        transition_matrix_file = 'transition_matrix_racetrack.npy'
        reward_matrix_file = 'reward_matrix_racetrack.npy'
        if not os.path.isfile(reward_matrix_file):
            print("Creating reward matrix")
            reward_matrix = self.reward_matrix()
            np.save(reward_matrix_file, reward_matrix)
        else:
            reward_matrix = np.load(reward_matrix_file)
        if not os.path.isfile(transition_matrix_file):
            print("Creating transition matrix")
            transition_matrix = self.transition_matrix(reward_matrix)
            np.save(transition_matrix_file, transition_matrix)
        else:
            transition_matrix = np.load(transition_matrix_file)

        terminal_states = np.zeros((self.num_states,))
        terminal_states[-1] = True

        initial_states = np.zeros((self.num_states,))
        initial_states[self._state_index((1, 6, 0, 0))] = 1

        self.tabular = Tabular(transition_matrix, reward_matrix, terminal_states, initial_states)

    def tabular_env(self):
        return TabularEnv(self.tabular)

    def one_hot_env(self):
        return OneHotEnv.from_tabular(self.tabular)

    def optimal_policy(self, discount_factor: float) -> np.ndarray:
        policy_file = f'racetrack_policy_{discount_factor}.npy'
        if os.path.isfile(policy_file):
            return np.load(policy_file)
        else:
            policy = value_iteration(self.tabular, 0.99)
            np.save(policy_file, policy)
            return policy


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

simple_grid2 = Gridworld(
    np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, G],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [S, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).T,
    np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).T
)


def _run():
    env = simple_grid1.tabular_env()
    policy = value_iteration(simple_grid1.tabular, 0.99)
    print(policy)
    state = env.reset()
    is_terminal = False
    while not is_terminal:
        print(type(state))
        print(simple_grid1.state_repr(state))
        state, reward, is_terminal, _ = env.step(policy[state])
    print(simple_grid1.state_repr(state))


def _solve_racetrack():
    racetrack = Racetrack()
    policy = value_iteration(racetrack.tabular, 0.99)
    np.save('racetrack_policy_0.99.npy', policy)


def _test_racetrack():
    policy = np.load('racetrack_policy_0.99.npy')
    racetrack = Racetrack()
    env = racetrack.tabular_env()
    state = env.reset()
    is_terminal = False
    score = 0
    while not is_terminal:
        state, reward, is_terminal, _ = env.step(policy[state])
        print(reward)
        score += reward
    print("===")
    print(score)


if __name__ == '__main__':
    _test_racetrack()
