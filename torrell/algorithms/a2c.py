from typing import Sequence, Callable, Optional, Type, Dict, Any

import actor.likelihood_ratio_gradient
import algorithms.algorithm
import critic.advantages
import critic.value_td
import environments.environment
import numpy as np
import policies.policy
import torch
import torch.optim.lr_scheduler
import trainers.data_collection
import trainers.synchronous

import torrell.core_algorithms.actor_critic


class A2C(algorithms.algorithm.Algorithm):
    def __init__(self, envs: Sequence[environments.environment.Environment], state_dim: int, action_dim: int,
                 value_network: torch.nn.Module, policy_model: policies.policy.PolicyModel,
                 policy_builder: Callable[[policies.policy.PolicyModel], policies.policy.Policy],
                 learning_rate: float, discount_factor: float, steps_per_agent: int=1,
                 batch_size: int=1, entropy_regularization: float=0, reward_log_smoothing: float=1, max_len: int=-1,
                 gradient_clipping: Optional[float]=None, lr_decay: float=-1,
                 advantage_type: Type[critic.advantages.AdvantageProvider]=critic.advantages.TDErrorAdvantageProvider,
                 advantage_provider_args: Optional[Dict[str, Any]]=None):
        if advantage_provider_args is None:
            advantage_provider_args = {}
        optimizer = torch.optim.RMSprop(set(value_network.parameters()) | set(policy_model.parameters),
                                        lr=learning_rate)

        pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(policy_model, entropy_regularization)
        tdv = critic.value_td.ValueTD(value_network, 1)
        tderror = advantage_type(tdv, **advantage_provider_args)
        self._actor_critic = torrell.core_algorithms.actor_critic.ActorCritic(
            optimizer, pg, tdv, tderror, gradient_clipping=gradient_clipping
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
        trainers_ = [
            trainers.data_collection.Trainer(env, self.policy, config)
            for env in envs
        ]
        self._synchronous = trainers.synchronous.ParallelAgents(trainers_, steps_per_agent, batch_size)
        super().__init__(trainers_, config, self.policy)

    def iterate(self, iteration: int):
        batch = self._synchronous.collect_transitions()
        if self._scheduler is not None:
            self._scheduler.step()
        self._actor_critic.iterate(iteration, batch)
