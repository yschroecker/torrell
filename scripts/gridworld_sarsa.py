import torch

import trainers.online_trainer
import environments.tabular
import actor.value
import critic.value_td
import critic.advantages


def _run():
    grid = environments.tabular.simple_grid1
    env = grid.one_hot_env()
    q_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    torch.nn.init.constant(q_network.weight, 1)
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.5)
    sarsa = critic.value_td.QValueTD(q_network, optimizer, 1)
    epsilon_greedy = actor.value.EpsilonGreedy(grid.num_actions, q_network, 0.2, 0.2, 0)
    config = trainers.online_trainer.TrainerConfig(
        env=env,
        num_actions=grid.num_actions,
        state_dim=grid.num_states,
        actor=epsilon_greedy,
        critic=sarsa,
        policy=epsilon_greedy,
        advantage_provider=critic.advantages.NoAdvantageProvider(),
        discount_factor=0.99,
        reward_log_smoothing=1
    )
    trainer = trainers.online_trainer.DiscreteNstepTrainer(config, batch_size=10)
    trainer.train(100)


if __name__ == '__main__':
    _run()
