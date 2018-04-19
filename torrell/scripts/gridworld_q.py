import torch

import environments.tabular
import algorithms.q_learning


def _run():
    grid = environments.tabular.simple_grid1
    env = grid.one_hot_env()
    test_env = grid.one_hot_env()
    q_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    torch.nn.init.constant(q_network.weight, 1)
    q_learner = algorithms.q_learning.QLearning(env, grid.num_states, grid.num_actions, q_network, 0.5, 0.99,
                                                max_len=1000)
    for iteration in q_learner.rl_eval_range(0, 1000, test_env, 100):
        q_learner.iterate(iteration)


if __name__ == '__main__':
    _run()
