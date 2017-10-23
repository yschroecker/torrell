import torch
import numpy as np

import critic.advantages
import critic.value_td
import environments.environment
import policies.softmax
import trainers.online_trainer
import optimization_strategies.simultaneous_gradient_descent
import actor.likelihood_ratio_gradient


def train(num_iterations: int, env: environments.environment.Environment, state_dim: int,
          value_network: torch.nn.Module, policy_network: torch.nn.Module, learning_rate: float, discount_factor: float,
          batch_size: int=1, entropy_regularization: float = 0, reward_log_smoothing: float=1, max_len: int=-1,
          evaluation_frequency: int=-1):
    optimizer = torch.optim.RMSprop(set(value_network.parameters()) | set(policy_network.parameters()),
                                    lr=learning_rate)

    softmax_policy = policies.softmax.SoftmaxPolicyModel(policy_network)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(softmax_policy, entropy_regularization)
    tdv = critic.value_td.ValueTD(value_network, 1)
    tderror = critic.advantages.TDErrorAdvantageProvider(tdv)
    strategy = optimization_strategies.simultaneous_gradient_descent.SimultaneousGradientDescent(
        optimizer, pg, tdv, tderror
    )
    config = trainers.online_trainer.TrainerConfig(
        state_dim=state_dim,
        policy_model=softmax_policy,
        policy_builder=policies.softmax.SoftmaxPolicy,
        optimization_strategy=strategy,
        discount_factor=discount_factor,
        max_len=max_len,
        reward_log_smoothing=reward_log_smoothing,
        evaluation_frequency=evaluation_frequency
    )
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, config, batch_size)
    trainer.train(num_iterations)
