from typing import Sequence

import numpy as np
import tqdm

from environments.tabular import simple_grid1 as grid
from scripts.tabular import policy, tdlsd

env = grid.tabular_env()
advice_eps = 0.2


def advice(state: int) -> int:
    y, x = grid.state_repr(state)
    if y > x:
        optimal_action = grid.RIGHT
    else:
        optimal_action = grid.UP

    if np.random.rand() < advice_eps:
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
                     learning_rate: float=0.0001) -> policy.BiasedTabularPolicy:
    supervised_policy = policy.BiasedTabularPolicy(env.num_states, env.num_actions)
    for _ in tqdm.trange(num_epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for batch in range(int(np.ceil(len(X)/batch_size))):
            batch_indices = indices[batch*batch_size:(batch + 1)*batch_size]
            gradient = supervised_policy.log_gradient(X[batch_indices], y[batch_indices])
            supervised_policy.parameters += learning_rate * gradient
    return supervised_policy


def train_sagil(X: np.ndarray, y: np.ndarray, num_epochs: int=500, batch_size: int=100,
                learning_rate: float=0.001) -> policy.BiasedTabularPolicy:
    tdlsd_estimator = tdlsd.TDLSD(grid.num_states, grid.num_actions, learning_rate=0.1, batch_size=100)
    sagil_policy = policy.BiasedTabularPolicy(env.num_states, env.num_actions)
    advice_dist = np.bincount(X, minlength=grid.num_states)/len(X)
    for epoch in tqdm.trange(num_epochs):
        states = []
        actions = []
        next_states = []
        for episode in range(100):
            is_terminal = False
            state = env.reset()
            t = 0
            while not is_terminal and t < 100:
                t += 1
                action = np.random.choice(env.num_actions, p=sagil_policy.probabilities(state)[0])
                states.append(state)
                actions.append(action)
                state, _, is_terminal, _ = env.step(action)
                next_states.append(state)
        tdlsd_estimator.update_gradient(np.array(states), np.array(actions), np.array(next_states), sagil_policy, 1)
        tdlsd_estimator.correct_gradient(np.array(states))

        pi_dist = np.bincount(states, minlength=grid.num_states)/len(X)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        if epoch > 100:
            for batch in range(int(np.ceil(len(X)/batch_size))):
                batch_indices = indices[batch*batch_size:(batch + 1)*batch_size]
                pi_theta = sagil_policy.probabilities(X[batch_indices])[np.arange(len(batch_indices)), y[batch_indices]]
                # pi_advice = advice_probabilities(X[batch_indices], y[batch_indices])
                # weights = pi_theta/pi_advice
                weights = pi_dist[X[batch_indices]]/advice_dist[X[batch_indices]]
                gradient = sagil_policy.log_gradient(X[batch_indices], y[batch_indices], weights)
                lsd = tdlsd_estimator.gradient(X[batch_indices])
                gradient += np.sum((weights * np.log(pi_theta))[:, np.newaxis, np.newaxis] * lsd, axis=0)
                sagil_policy.parameters += learning_rate * gradient

    return sagil_policy


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
            states.append(state)
            actions.append(action)
            state, _, is_terminal, _ = env.step(action)
    states = np.array(states)
    actions = np.array(actions)

    # noinspection PyUnusedLocal
    supervised_policy = train_supervised(states, actions)
    advice_policy = train_sagil(states, actions)
    lengths = []
    for _ in tqdm.trange(1000):
        is_terminal = False
        state = env.reset()
        length = 0
        while not is_terminal and length < 1000:
            action = np.random.choice(env.num_actions, p=advice_policy.probabilities(state)[0])
            state, _, is_terminal, _ = env.step(action)
            length += 1
        lengths.append(length)
    print(np.mean(lengths))


if __name__ == '__main__':
    _run()
