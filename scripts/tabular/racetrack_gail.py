import environments.tabular
import tqdm
import sklearn.linear_model
import scripts.tabular.policy as policy

import numpy as np


def _run():
    num_demo_episodes = 100
    racetrack = environments.tabular.Racetrack()
    optimal = racetrack.optimal_policy(0.99)

    X = []
    env = racetrack.tabular_env()
    for episode in range(num_demo_episodes):
        is_terminal = False
        state = env.reset()
        while not is_terminal:
            action = optimal[state]
            X.append([state, action])
            state, _, is_terminal, _ = env.step(action)

    current_policy = policy.TabularPolicy(racetrack.num_states, racetrack.num_actions, False)
    lr_model = sklearn.linear_model.LogisticRegression()
    episodes = []
    Q = np.zeros((racetrack.num_states * racetrack.num_actions,))
    for iteration in tqdm.trange(5000):
        episode_sa = []
        state = env.reset()
        for t in range(5000):
            action = np.random.choice(np.arange(racetrack.num_actions), p=current_policy.probabilities(state)[0])
            episode_sa.append([state, action])
            state, _, is_terminal, _ = env.step(action)
            if is_terminal:
                break
        episodes.append(episode_sa)

        inputs = np.concatenate([X, np.concatenate(episodes[-num_demo_episodes:])])
        labels = np.concatenate([np.zeros(len(X)), np.ones(len(inputs) - len(X))])
        lr_model.fit(inputs, labels)


if __name__ == '__main__':
    _run()