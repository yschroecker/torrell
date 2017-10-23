import torch

import environments.tabular
import algorithms.q_learning


def _run():
    grid = environments.tabular.simple_grid1
    env = grid.one_hot_env()
    q_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    torch.nn.init.constant(q_network.weight, 1)
    algorithms.q_learning.train(1000, env, grid.num_states, grid.num_actions, q_network, 0.5, 0.99)


if __name__ == '__main__':
    _run()
