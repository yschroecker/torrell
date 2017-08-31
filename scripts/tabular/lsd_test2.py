import numpy as np

import environments.tabular
from scripts.tabular import policy

num_states = 5


def build_chain():
    transition_matrix = np.zeros((num_states * 2, num_states))
    for state in range(num_states):
        transition_matrix[state * 2, state] = 1
        transition_matrix[state * 2 + 1, (state + 1) % num_states] = 1

    reward_matrix = np.zeros((num_states * 2, num_states))

    terminal_states = np.zeros((num_states,))
    initial_states = np.zeros((num_states,))
    initial_states[num_states//2] = 1

    tabular = environments.tabular.Tabular(transition_matrix, reward_matrix, terminal_states, initial_states)
    return tabular


def _run():
    chain = build_chain()
    current_policy = policy.BiasedTabularPolicy(num_states, chain.num_actions)
    current_policy.parameters = np.array([[1, 1], [3, 1], [1, 1], [3, 2], [2, 2.5], [0, 0]])
    stationary_distribution = chain.stationary_state_distribution(current_policy.matrix)
    print(stationary_distribution)
    env = environments.tabular.TabularEnv(chain)
    states = []
    state = env.reset()
    for _ in range(10000):
        action = np.random.choice(2, p=current_policy.probabilities([state])[0])
        state, _, _, _ = env.step(action)
        states.append(state)
    print(np.bincount(states)/len(states))
    lsd = chain.log_stationary_derivative(current_policy.matrix, current_policy.log_gradient_matrix,
                                          stationary_distribution)
    print(lsd[:, 0, 1])


if __name__ == '__main__':
    _run()
