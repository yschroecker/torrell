import environments.tabular
import numpy as np

from torrell.scripts.tabular import policy

num_states = 5
num_actions = 2


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
    current_policy = policy.TabularPolicy(num_states, chain.num_actions)
    current_policy.parameters = np.array([[1, 1], [3, 1], [1, 1], [3, 2], [2, 2.5], [0, 0]])
    stationary_distribution = chain.stationary_state_distribution(current_policy.matrix)
    print(stationary_distribution)
    env = environments.tabular.TabularEnv(chain)
    actions = []
    state = env.reset()
    states = [state]
    for _ in range(10000):
        action = np.random.choice(2, p=current_policy.probabilities([state])[0])
        actions.append(action)
        state, _, _, _ = env.step(action)
        states.append(state)
    print(np.bincount(states)/len(states))
    lsd = chain.log_stationary_derivative(current_policy.matrix, current_policy.log_gradient_matrix,
                                          stationary_distribution)
    print(lsd[:, 0, 1])


    values = np.zeros((num_states,))
    mu = 0
    lr = 0.01
    log_probabilities = np.log(current_policy.probabilities(np.arange(num_states)))
    for state, action, next_state in zip(states[:-1], actions[:-1], states[1:]):
        logpi = log_probabilities[state][action]
        mu += lr * (logpi - mu)
        values[next_state] += lr * (logpi - mu + values[state] - values[next_state])

    reverse_rewards = np.zeros((num_states, num_actions))# det case
    for state, action, next_state in zip(states[:-1], actions[:-1], states[1:]):
        reverse_rewards[state, action] = values[next_state] - values[state]





    '''
    q_values = np.zeros((num_states, num_actions))
    for state, action, next_state in zip(states[:-1], actions[:-1], states[1:]):
        q_values[state, action] = values[next_state]
    print(q_values)

    lsd01 = np.zeros((num_states,))
    for state in range(num_states):
        pi = current_policy.probabilities([state])[0]
        for action in range(num_actions):
            probability = pi[action]
            log_probability_gradients = current_policy.log_gradient([state], [action])

            lsd01[state] += probability * log_probability_gradients[0, 1] * q_values[state, action]
    print(lsd01)
    '''


if __name__ == '__main__':
    _run()
