from typing import Sequence

import numpy as np
import sklearn.preprocessing
import sklearn.linear_model
import tqdm

from environments.tabular import simple_grid1 as grid
from scripts.tabular import policy

env = grid.tabular_env()
advice_eps = 0.2


def advice(state: int, eps: float=advice_eps) -> int:
    x, y = grid.state_repr(state)
    if y > x:
        optimal_action = grid.RIGHT
    else:
        optimal_action = grid.UP

    if np.random.rand() < eps:
        return np.random.choice(env.num_actions)
    else:
        return optimal_action


def advice_probabilities(states: Sequence[int], actions: Sequence[int]) -> np.ndarray:
    result = []
    for state, action in zip(states, actions):
        if action == advice(state):
            result.append((1 - advice_eps) + advice_eps / grid.num_actions)
        else:
            result.append(advice_eps/(grid.num_actions - 1))
    return np.array(result)


def train_supervised(X: np.ndarray, y: np.ndarray, num_epochs: int=100, batch_size: int=100,
                     learning_rate: float=0.001) -> policy.BiasedTabularPolicy:
    supervised_policy = policy.BiasedTabularPolicy(env.num_states, env.num_actions)
    for _ in tqdm.trange(num_epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for batch in range(int(np.ceil(len(X)/batch_size))):
            batch_indices = indices[batch*batch_size:(batch + 1)*batch_size]
            gradient = supervised_policy.log_gradient(X[batch_indices], y[batch_indices])
            supervised_policy.parameters += learning_rate * gradient
    return supervised_policy


def train_sagil(X: np.ndarray, y: np.ndarray, original_policy: policy.BiasedTabularPolicy, num_epochs: int=100,
                batch_size: int=100, learning_rate: float=0.001,
                approximate_weights: bool=False) -> policy.BiasedTabularPolicy:
    sagil_policy = policy.BiasedTabularPolicy(env.num_states, env.num_actions)
    advice_state_dist = grid.tabular.stationary_state_distribution(original_policy.matrix)
    for _ in tqdm.trange(num_epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for batch in range(int(np.ceil(len(X)/batch_size))):
            batch_indices = indices[batch*batch_size:(batch + 1)*batch_size]
            state_distribution = grid.tabular.stationary_state_distribution(sagil_policy.matrix)
            gradient_function = grid.tabular.log_stationary_derivative(
                sagil_policy.matrix, sagil_policy.log_gradient_matrix, state_distribution
            )

            if approximate_weights:
                weights = sagil_policy.probabilities(X[batch_indices])[np.arange(len(batch_indices)),
                                                                       y[batch_indices]] / \
                          original_policy.probabilities(X[batch_indices])[np.arange(len(batch_indices)),
                                                                          y[batch_indices]]
            else:
                weights = state_distribution[X[batch_indices]]/advice_state_dist[X[batch_indices]]
            weights /= len(weights)
            pi_theta = sagil_policy.probabilities(X[batch_indices])[np.arange(len(batch_indices)), y[batch_indices]]
            gradient = sagil_policy.log_gradient(X[batch_indices], y[batch_indices], weights)
            lsd = gradient_function[X[batch_indices]]
            gradient += np.sum((weights * np.log(pi_theta))[:, np.newaxis, np.newaxis] * lsd, axis=0)
            sagil_policy.parameters += learning_rate * gradient

    return sagil_policy


def test_policy(current_policy: policy.PolicyBase) -> float:
    lengths = []
    for _ in tqdm.trange(1000):
        is_terminal = False
        state = env.reset()
        length = 0
        while not is_terminal and length < 1000:
            action = np.random.choice(env.num_actions, p=current_policy.probabilities(state)[0])
            state, _, is_terminal, _ = env.step(action)
            length += 1
        lengths.append(length)
    return np.asscalar(np.mean(lengths))


class SklearnModel(policy.PolicyBase):
    def __init__(self, states: Sequence[int], actions: Sequence[int]):
        self._state_encoder = sklearn.preprocessing.OneHotEncoder(n_values=env.num_states)
        self._action_encoder = sklearn.preprocessing.LabelEncoder()
        self._model = sklearn.linear_model.LogisticRegression(C=1000, fit_intercept=True)
        transformed_states = self._state_encoder.fit_transform(np.atleast_2d(states).T)
        self._model.fit(transformed_states, self._action_encoder.fit_transform(actions))

    def probabilities(self, state: int) -> np.ndarray:
        class_probabilities = self._model.predict_proba(self._state_encoder.transform(np.atleast_2d(state).T))[0]
        action_probabilities = np.zeros((grid.num_actions,))
        for i, class_probability in enumerate(class_probabilities):
            action_probabilities[self._action_encoder.inverse_transform([i])[0]] = class_probability
        return np.array([action_probabilities])


def _run():
    initial_policy = policy.BiasedTabularPolicy(env.num_states, env.num_actions)
    initial_policy.parameters[policy.BIAS, grid.RIGHT] = 5 + np.log(0.5)
    initial_policy.parameters[policy.BIAS, grid.UP] = 5

    # d = grid.tabular.stationary_state_distribution(initial_policy.matrix)
    print()

    states = []
    actions = []
    for episode in range(1000):
        is_terminal = False
        state = env.reset()
        while not is_terminal:
            action = np.random.choice(env.num_actions, p=initial_policy.probabilities(state)[0])
            advice_action = advice(state)
            states.append(state)
            actions.append(advice_action)
            state, _, is_terminal, _ = env.step(action)
    states = np.array(states)
    actions = np.array(actions)

    sklearn_policy = SklearnModel(states, actions)
    sklearn_length = test_policy(sklearn_policy)
    print(f"sklearn length: {sklearn_length}")

    # noinspection PyUnusedLocal
    supervised_policy = train_supervised(states, actions)
    supervised_length = test_policy(supervised_policy)
    print(f"supervised length: {supervised_length}")
    advice_policy = train_sagil(states, actions, initial_policy)
    advice_length = test_policy(advice_policy)
    print(f"advice length: {advice_length}")


if __name__ == '__main__':
    _run()
