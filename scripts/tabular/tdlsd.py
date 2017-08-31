from scripts.tabular import policy

import numpy as np


class TDLSD:
    def __init__(self, num_states: int, num_actions: int, learning_rate: float, batch_size: int,
                 discount_factor: float = 1):
        self._gradient_target = np.zeros((num_states, num_states + 1, num_actions))
        self._gradient = np.zeros((num_states, num_states + 1, num_actions))
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._discount_factor = discount_factor

    def _update_gradient_batch(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
                               current_policy: policy.BiasedTabularPolicy):
        policy_derivative = current_policy.log_gradient(states, actions)
        update = (policy_derivative + self._discount_factor * self._gradient_target[states, :, :] -
                  self._gradient_target[next_states, :, :])
        self._gradient_target[next_states, :, :] += self._learning_rate * update

    def update_gradient(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
                        current_policy: policy.BiasedTabularPolicy, num_epochs: int):
        num_batches = int(np.ceil(len(actions)/self._batch_size))
        indices = np.arange(len(actions))
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for batch in range(num_batches):
                batch_indices = indices[batch * self._batch_size:(batch + 1) * self._batch_size]
                self._update_gradient_batch(states[batch_indices], actions[batch_indices], next_states[batch_indices],
                                            current_policy)

    def correct_gradient(self, states):
        self._gradient = self._gradient_target - np.mean(self._gradient_target[states, :, :], axis=0, keepdims=True)

    def gradient(self, states):
        return self._gradient[states, :, :]
