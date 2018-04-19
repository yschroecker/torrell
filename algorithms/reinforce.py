from typing import Sequence, Callable, Optional

import torch
import torch.optim.lr_scheduler
import numpy as np

import algorithms.algorithm
import critic.advantages
import critic.value_td
import environments.environment
import policies.policy
import trainers.data_collection
import trainers.synchronous
import core_algorithms.actor_only
import actor.likelihood_ratio_gradient


class Reinforce(algorithms.algorithm.Algorithm):
    def __init__(self, env: environments.environment.Environment, state_dim: int, action_dim: int,
                 policy_model: policies.policy.PolicyModel,
                 policy_builder: Callable[[policies.policy.PolicyModel], policies.policy.Policy],
                 learning_rate: float, mean_reward_update_rate: float, discount_factor: float, num_trajectories: int=1,
                 entropy_regularization: float=0, reward_log_smoothing: float=1, max_len: int=-1,
                 gradient_clipping: Optional[float]=None, lr_decay: float=-1, optimizer_type: str='rms'):
        self._num_trajectories = num_trajectories
        if optimizer_type == 'rms':
            optimizer = torch.optim.RMSprop(policy_model.parameters, lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(policy_model.parameters, lr=learning_rate)

        pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(policy_model, entropy_regularization)
        mc_returns = critic.advantages.MeanRewardMonteCarlo(mean_reward_update_rate)
        self._actor_only = core_algorithms.actor_only.ActorOnly(
            optimizer, pg, mc_returns, gradient_clipping=gradient_clipping
        )
        if lr_decay > 0:
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda e: 1/np.sqrt(1 + (1 - lr_decay) * e)
            )
        else:
            self._scheduler = None
        config = trainers.data_collection.TrainerConfig(
            state_dim, discount_factor, reward_log_smoothing, action_dim, max_len
        )
        self.policy = policy_builder(policy_model)
        self._trainer = trainers.data_collection.Trainer(env, self.policy, config)
        super().__init__([self._trainer], config, self.policy)

    def iterate(self, iteration: int):
        batch = trainers.data_collection.collect_batch(self._trainer, -1, self._num_trajectories)
        if self._scheduler is not None:
            self._scheduler.step()
        self._actor_only.iterate(iteration, batch)
