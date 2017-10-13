import environments.tabular
import tqdm
import sklearn.linear_model
from scripts.tabular import policy, supervised

import numpy as np


def _run():
    num_demo_episodes = 100
    racetrack = environments.tabular.Racetrack()
    optimal = racetrack.optimal_policy(0.99)

    X = []
    env = racetrack.tabular_env()
    for episode in tqdm.trange(num_demo_episodes):
        is_terminal = False
        state = env.reset()
        while not is_terminal:
            action = optimal[state]
            X.append([state, action])
            state, _, is_terminal, _ = env.step(action)
    X = np.array(X)

    # current_policy = policy.TabularPolicy(racetrack.num_states, racetrack.num_actions, False)
    current_policy = supervised.train_supervised(env, X[:,0], X[:, 1], 10000, 32, 1e-4, 1)
    lr_model = sklearn.linear_model.LogisticRegression()
    episodes = []
    V = np.zeros((racetrack.num_states))
    trange = tqdm.trange(5000)
    ema = 0
    for iteration in trange:
        episode_sa = []
        state = env.reset()
        cumulative_score = 0
        for t in range(500):
            action = np.random.choice(np.arange(racetrack.num_actions), p=current_policy.probabilities(state)[0])
            episode_sa.append([state, action])
            next_state, score, is_terminal, _ = env.step(action)
            if iteration == 4999:
                print(score)
            cumulative_score += score
            if iteration > 1:
                reward = -lr_model.predict_log_proba(np.array([[state, action]]))[0][1]
                V[state] += 0.1 * (reward + 0.99*V[next_state] - V[state])
                current_policy.parameters += 0.1 * (current_policy.log_gradient([state], [action]) * (V[next_state] + reward
                    - V[state]))
            #if t == 100:
                #print(current_policy.probabilities(state))
            state = next_state
            if is_terminal:
                #print(f"terminating after {t}")
                #print(score)
                break
        if iteration % 100 == 0:
            print()
        ema = 0.9*ema + 0.1 * cumulative_score
        trange.set_description(f"score: {ema}")
        episodes.append(episode_sa)

        inputs = np.concatenate([X, np.concatenate(episodes[-num_demo_episodes:])])
        labels = np.concatenate([np.zeros(len(X)), np.ones(len(inputs) - len(X))])
        lr_model.fit(inputs, labels)


if __name__ == '__main__':
    _run()
