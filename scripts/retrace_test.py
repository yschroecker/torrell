import torch
import numpy as np

import environments.tabular
import critic.retrace
import critic.value_td
import policies.softmax
import scripts.tabular.policy
import torch_util
import data
import tqdm


def _run():
    grid = environments.tabular.simple_grid1
    v_network = torch.nn.Linear(grid.num_states, 1, bias=False)
    torch.nn.init.constant(v_network.weight, 0)

    uniform_policy_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    torch.nn.init.constant(uniform_policy_network.weight, 0)
    sample_policy = policies.softmax.SoftmaxPolicyModel(uniform_policy_network)

    oracle_policy = environments.tabular.value_iteration(grid.tabular, 0.9, 0.001)
    greedy_proba = 0.5
    eval_policy_matrix = np.zeros((grid.num_states, grid.num_actions))
    eval_policy_matrix[np.arange(grid.num_states), oracle_policy] = np.log(greedy_proba) - np.log((1-greedy_proba)/(grid.num_actions-1))
    eval_tabular_policy = scripts.tabular.policy.TabularPolicy(grid.num_states, grid.num_actions, biased=False)
    eval_tabular_policy.parameters = eval_policy_matrix
    print(environments.tabular.evaluate_probabilistic(eval_tabular_policy.matrix, grid.tabular, 0.9))
    eval_policy_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    eval_policy_network.weight.data = torch_util.load_input(False, eval_policy_matrix.T.astype(np.float32))
    eval_policy = policies.softmax.SoftmaxPolicyModel(eval_policy_network)

    tdv = critic.retrace.Retrace(v_network, sample_policy, eval_policy, 0.9, 1)
    # tdv = critic.value_td.ValueTD(v_network, 1)
    sample_sample_policy = policies.softmax.SoftmaxPolicy(sample_policy)
    env = grid.one_hot_env()
    state = env.reset()
    action = sample_sample_policy.sample(state, 0)
    optimizer = torch.optim.SGD(v_network.parameters(), 0.05)
    for _ in tqdm.trange(10000):
        states = [state]
        actions = [action]
        discount_weights = []
        rewards = []
        is_terminal = False
        for t in range(4):
            state, reward, is_terminal, _ = env.step(action)
            action = sample_sample_policy.sample(state, 0)
            states.append(np.array(state, dtype=np.float32))
            actions.append(action)
            rewards.append(reward)
            discount_weights.append(0.9**(t + 1))
            if is_terminal:
                state = env.reset()
                action = sample_sample_policy.sample(state, 0)
                break
        sequences = [data.RLTransitionSequence(
            rewards=np.array(rewards, np.float32)[j:],
            transition_sequence=data.TransitionSequence(
                states=np.array(states[j:], dtype=np.float32),
                actions=np.array(actions[j:], np.int32),
                is_terminal=np.array([is_terminal], dtype=np.float32),
                discount_weights=np.array((discount_weights if j == 0 else discount_weights[:-j]), dtype=np.float32)
            )
        ) for j in range(len(rewards))]
        batch = data.Batch(sequences)
        loss = tdv.update_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(next(v_network.parameters()).data.numpy())


if __name__ == '__main__':
    _run()
    # algorithms.sarsa.train(1000, env, grid.num_states, grid.num_actions, q_network, 0.5, 0.99)
