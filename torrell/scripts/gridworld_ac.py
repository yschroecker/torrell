import actor.likelihood_ratio_gradient
import critic.advantages
import critic.value_td
import environments.tabular
import policies.softmax
import torch
import trainers.online_trainer

import torrell.core_algorithms.actor_critic


def _run():
    grid = environments.tabular.simple_grid1
    env = grid.one_hot_env()
    v_network = torch.nn.Linear(grid.num_states, 1, bias=False)
    policy_network = torch.nn.Linear(grid.num_states, grid.num_actions, bias=False)
    #v_network.cuda()
    #policy_network.cuda()
    optimizer = torch.optim.SGD(set(v_network.parameters()) | set(policy_network.parameters()), lr=0.5)
    tdv = critic.value_td.ValueTD(v_network, 1)
    softmax_policy = policies.softmax.SoftmaxPolicyModel(policy_network)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(softmax_policy)

    strategy = torrell.core_algorithms.actor_critic.ActorCritic(
        optimizer, pg, tdv, critic.advantages.TDErrorAdvantageProvider(tdv)
    )

    config = trainers.online_trainer.TrainerConfig(
        state_dim=grid.num_states,
        optimization_strategy=strategy,
        policy_model=softmax_policy,
        policy_builder=policies.softmax.SoftmaxPolicy,
        discount_factor=0.99,
        reward_log_smoothing=0.1,
        evaluation_frequency=50,
        max_len=10000,
        hooks=[]
    )
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, config, batch_size=32)
    trainer.train(10000)


if __name__ == '__main__':
    _run()
