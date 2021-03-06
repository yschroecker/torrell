from abc import ABCMeta
from typing import Sequence, Optional

import numpy as np

BIAS = -1


class PolicyBase(metaclass=ABCMeta):
    def probabilities(self, state: Sequence[int]) -> np.ndarray:
        pass


class TabularPolicy(PolicyBase):
    def __init__(self, num_states: int, num_actions: int, biased: bool=True):
        self._num_states = num_states
        self._num_actions = num_actions
        self._biased = biased
        self.parameters = np.zeros((num_states + (1 if biased else 0), num_actions))

    def scores(self, state: Sequence[int]) -> np.ndarray:
        score = self.parameters[state, :]
        if self._biased:
            score += self.parameters[-1, :]
        return score

    def sample(self, state: Sequence[int]) -> Sequence[int]:
        probabilities = self.probabilities(state)
        return [np.random.choice(self._num_actions, p=state_probabilities) for state_probabilities in probabilities]

    def sample_one(self, state: int):
        return self.sample([state])[0]

    def probabilities(self, state: Sequence[int]) -> np.ndarray:
        state = np.atleast_1d(state)
        exp_scores = np.exp(self.scores(state))
        return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    def log_gradient(self, states: Sequence[int], actions: Sequence[int],
                     weights: Optional[Sequence[float]]=None) -> np.ndarray:
        gradient = np.zeros_like(self.parameters)
        probabilities = self.probabilities(states)
        if weights is None:
            weights = np.ones_like(actions)
        for state, action, pi, weight in zip(states, actions, probabilities, weights):
            gradient[state, :] -= pi * weight
            gradient[state, action] += 1 * weight
            if self._biased:
                gradient[BIAS, :] -= pi * weight
                gradient[BIAS, action] += 1 * weight

        return gradient

    @property
    def matrix(self) -> np.ndarray:
        result = np.zeros((self._num_states, self._num_states * self._num_actions))
        for state in range(self._num_states):
            result[state, state * self._num_actions:(state + 1) * self._num_actions] = self.probabilities([state])[0]
        return result

    @property
    def log_gradient_matrix(self) -> np.ndarray:
        result = np.zeros((self._num_states * self._num_actions,) + self.parameters.shape)
        for sa in range(self._num_states * self._num_actions):
            state = sa // self._num_actions
            action = sa % self._num_actions
            result[sa, ...] = self.log_gradient([state], [action])
        return result


if __name__ == '__main__':
    policy = TabularPolicy(3, 2)
    print(policy.probabilities([0, 1]))
    print(policy.log_gradient([0, 1, 0], [1, 0, 0]))
    policy = TabularPolicy(3, 2, False)
    print(policy.probabilities([0, 1]))
    print(policy.log_gradient([0, 1, 0], [1, 0, 0]))
