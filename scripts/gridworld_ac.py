import torch

import trainers.online_trainer
import environments.tabular
import actor.likelihood_ratio_gradient
import policies.softmax
import critic.value_td
import critic.advantages


def _run():
    grid = environments.tabular.simple_grid1
    env = grid.one_hot_env()
    v_network = torch.nn.Linear(grid.num_states, 1, bias=False)
    policy_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    critic_optimizer = torch.optim.SGD(v_network.parameters(), lr=0.5)
    policy_optimizer = torch.optim.SGD(policy_network.parameters(), lr=5)
    tdv = critic.value_td.ValueTD(v_network, critic_optimizer, 1)
    softmax_policy = policies.softmax.SoftmaxPolicy(policy_network)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(softmax_policy, policy_optimizer)
    td_error = critic.advantages.TDErrorAdvantageProvider(tdv)

    config = trainers.online_trainer.TrainerConfig(
        env=env,
        num_actions=grid.num_actions,
        state_dim=grid.num_states,
        actor=pg,
        critic=tdv,
        policy=softmax_policy,
        advantage_provider=td_error,
        discount_factor=0.99,
        reward_log_smoothing=1
    )
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(config, batch_size=32)
    trainer.train(1000)


if __name__ == '__main__':
    _run()
