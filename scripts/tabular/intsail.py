from typing import NewType, Tuple, Sequence

import numpy as np

import environments.tabular

ActionDistribution = NewType('ActionDistribution', Tuple[float, float])

num_actions = 2
num_states = 7


def chain():
    transition_matrix = np.zeros((num_states * 2, num_states))
    for state in range(num_states):
        transition_matrix[state * 2, num_states//2 if state == 0 else state - 1] = 1
        transition_matrix[state * 2 + 1, num_states//2 if state == num_states - 1 else state + 1] = 1

    reward_matrix = np.zeros((num_states * 2, num_states))

    terminal_states = np.zeros((num_states,))
    initial_states = np.zeros((num_states,))
    initial_states[num_states//2] = 1

    tabular = environments.tabular.Tabular(transition_matrix, reward_matrix, terminal_states, initial_states)
    return environments.tabular.TabularEnv(tabular)


def policy(_: int) -> ActionDistribution:
    return 1/3, 2/3


def simulate(num_steps: int) -> Tuple[Sequence[int], Sequence[float], Sequence[float]]:
    env = chain()

    states = []
    action_probabilities = []

    step_weights = []
    state = env.reset()
    for t in range(num_steps):
        action_distribution = policy(state)
        # sample_distribution = [1/3, 2/3]
        action = np.random.choice(num_actions, p=action_distribution)
        # step_weights.append(1)
        # action = np.random.choice(num_actions, p=sample_distribution)
        step_weights.append(0.5/action_distribution[action])

        states.append(state)
        action_probabilities.append(action_distribution[action])

        state, reward, is_terminal, _ = env.step(action)

    return states, action_probabilities, step_weights


def empirical_state_distribution(states: Sequence[int]) -> Sequence[float]:
    return np.bincount(states)/len(states)


def int_sail_estimate(states: Sequence[int], action_probabilities: Sequence[float], weights: Sequence[float],
                      learning_rate: float=0.1) \
        -> Sequence[float]:
    value = np.ones((num_states,))
    # mu = 0
    mu = (np.log(action_probabilities)).mean()
    # e = 0
    for state, action_probability, next_state, weight in zip(states[:-1], action_probabilities, states[1:], weights):
        log_pi = np.log(action_probability)
        # mu = (1 - learning_rate) * mu + learning_rate*log_pi
        value[next_state] += learning_rate * (log_pi - mu + value[state] - value[next_state]) / action_probability
    state_weights = np.exp(value)
    return state_weights/np.sum(state_weights)


def run():
    sample_states, sample_action_probabilities, weights = simulate(100000)
    empirical_dist = empirical_state_distribution(sample_states)
    print(empirical_dist)
    sail_dist = int_sail_estimate(sample_states, sample_action_probabilities, weights, 0.001)
    print(sail_dist)


if __name__ == '__main__':
    run()
