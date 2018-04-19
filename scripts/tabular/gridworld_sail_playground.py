import numpy as np
import tqdm

import environments.tabular
import scripts.tabular.policy


grid = environments.tabular.noisy_simple_grid1(0.5)


def demonstrations():
    Xy = [(grid.state_index(0, 4), 1), (grid.state_index(1, 4), 2), (grid.state_index(1, 3), 1),
          (grid.state_index(2, 3), 2), (grid.state_index(2, 2), 1), (grid.state_index(3, 2), 2),
          (grid.state_index(3, 1), 1), (grid.state_index(4, 1), 2)]
    X = [x for x, y in Xy]
    y = [y for x, y in Xy]
    return X, y


def estimate_gradient0(X, y, env: environments.tabular.TabularEnv, policy: scripts.tabular.policy.TabularPolicy) -> \
        np.ndarray:
    return policy.log_gradient(X, y)


def estimate_gradient1(X, y, env: environments.tabular.TabularEnv, policy: scripts.tabular.policy.TabularPolicy) -> \
        np.ndarray:
    d = grid.tabular.stationary_state_distribution(policy.matrix)
    gradient = grid.tabular.log_stationary_derivative(policy.matrix, policy.log_gradient_matrix, d)
    return np.sum(gradient[X, :], axis=0) + policy.log_gradient(X, y)


def estimate_gradient2(X, y, env: environments.tabular.TabularEnv, policy: scripts.tabular.policy.TabularPolicy) -> \
        np.ndarray:
    objective = np.zeros((env.num_states,))
    mu = 0

    state = env.reset()
    probabilities = policy.probabilities(np.arange(env.num_states))
    for _ in range(1000):
        action_probabilities = probabilities[state]
        action = np.random.choice(env.num_actions, p=action_probabilities)
        reward = np.log(action_probabilities[action])
        next_state, _, is_terminal, _ = env.step(action)

        mu = 0.95 * mu + 0.05 * reward
        objective[next_state] += 0.05 * (reward + objective[state] - objective[next_state])

        if is_terminal:
            state = env.reset()
        else:
            state = next_state

    log_gradients = policy.log_gradient_matrix
    gradient = np.zeros((env.num_states, env.num_actions))
    for i in range(len(y) - 2, -1, -1):
        gradient -= log_gradients[X[i] * env.num_actions + y[i], ...] * (objective[X[i]] + probabilities[X[i]][y[i]])
    return gradient + policy.log_gradient(X, y)

    # d = grid.tabular.stationary_state_distribution(policy.matrix)
    # #transition_matrix: np.ndarray, reward_matrix: np.ndarray, terminal_states: np.ndarray, initial_states: np.ndarray
    # initial_states = np.ones((env.num_states,))/env.num_states
    # terminal_states = np.zeros_like(initial_states)
    # transition_matrix = grid.tabular.reverse_transition_matrix(policy.matrix, d)
    #
    # reward_vector = np.
    # reverse_tab = environments.tabular.Tabular(transition_matrix, reward_matrix, )
    #
    # objective = environments.tabular.evaluate_probabilistic(policy.matrix, reverse_tab, 0.999)/1000


def _run():
    X, y = demonstrations()

    methods = [estimate_gradient2]
    for estimate_gradient in methods:
        policy = scripts.tabular.policy.TabularPolicy(grid.num_states, grid.num_actions, biased=False)
        env = grid.tabular_env()
        for iteration in tqdm.trange(2000):
            gradient = estimate_gradient(X, y, env, policy)
            policy.parameters += 0.05 * gradient

            if iteration % 20 == 0:
                values = environments.tabular.evaluate_probabilistic(policy.matrix, grid.tabular, 0.99)
                print(values[grid.state_index(0, 4)])



if __name__ == '__main__':
    _run()
