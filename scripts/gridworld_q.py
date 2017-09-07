import torch

import trainers.online_trainer
import environments.tabular
import actor.value
import critic.control.q_learning


def _run():
    grid = environments.tabular.simple_grid1
    env = grid.one_hot_env()
    q_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    torch.nn.init.constant(q_network.weight, 1)
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.5)
    q_learner = critic.control.q_learning.DiscreteQLearning(q_network, optimizer, 1)
    epsilon_greedy = actor.value.EpsilonGreedy(grid.num_actions, q_network, 0, 0, 0)
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, grid.num_actions, epsilon_greedy, q_learner, 0.99)
    trainer.train(1000)


if __name__ == '__main__':
    _run()
