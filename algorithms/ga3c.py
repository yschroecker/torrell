from typing import Sequence, Callable, Optional, Tuple

import torch
import numpy as np

import critic.advantages
import critic.value_td
import environments.environment
import policies.policy
import trainers.online_trainer
import trainers.synchronous
import optimization_strategies.simultaneous_gradient_descent
import actor.likelihood_ratio_gradient


def train(num_iterations: int, envs: Sequence[environments.environment.Environment], state_dim: int,
          value_network: torch.nn.Module, policy_model: policies.policy.PolicyModel,
          policy_builder: Callable[[policies.policy.PolicyModel], policies.policy.Policy],
          learning_rate: float, discount_factor: float, look_ahead: int=1,
          batch_size: int=1, entropy_regularization: float=0, reward_log_smoothing: float=1, max_len: int=-1,
          evaluation_frequency: int=-1, gradient_clipping: Optional[float]=None,
          reward_clipping: Tuple[float, float]=(-np.float('inf'), np.float('inf'))):
    optimizer = torch.optim.RMSprop(set(value_network.parameters()) | set(policy_model.parameters),
                                    lr=learning_rate)

    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(policy_model, entropy_regularization)
    tdv = critic.value_td.ValueTD(value_network, 1)
    tderror = critic.advantages.TDErrorAdvantageProvider(tdv)
    strategy = optimization_strategies.simultaneous_gradient_descent.SimultaneousGradientDescent(
        optimizer, pg, tdv, tderror, gradient_clipping=gradient_clipping
    )
    config = trainers.online_trainer.TrainerConfig(
        state_dim=state_dim,
        policy_model=policy_model,
        policy_builder=policy_builder,
        optimization_strategy=strategy,
        discount_factor=discount_factor,
        max_len=max_len,
        reward_log_smoothing=reward_log_smoothing,
        evaluation_frequency=evaluation_frequency,
        reward_clipping=reward_clipping
    )
    trainer = trainers.synchronous.SynchronizedDiscreteNstepTrainer(envs, config, look_ahead, batch_size)
    trainer.train(num_iterations)
