import torch
import numpy as np

import critic.advantages
import critic.value_td
import environments.environment
import policies.value
import trainers.online_trainer
import optimization_strategies.critic_only


def train(num_iterations: int, env: environments.environment.Environment, state_dim: int, num_actions: int,
          q_network: torch.nn.Module, learning_rate: float, discount_factor: float,
          initial_epsilon: float=0, final_epsilon: float=0, epsilon_decay_rate: float=1, reward_log_smoothing: float=1,
          max_len: int=-1, evaluation_frequency: int=-1):
    optimizer = torch.optim.SGD(q_network.parameters(), lr=learning_rate)
    q_learner = critic.value_td.QValueTD(q_network, 1)
    epsilon_greedy = policies.value.ValuePolicyModel(num_actions, q_network)
    strategy = optimization_strategies.critic_only.CriticOnly(optimizer, q_learner)
    config = trainers.online_trainer.TrainerConfig(
        state_dim=state_dim,
        policy_model=epsilon_greedy,
        policy_builder=policies.value.epsilon_greedy(initial_epsilon, final_epsilon, epsilon_decay_rate),
        optimization_strategy=strategy,
        discount_factor=discount_factor,
        reward_log_smoothing=reward_log_smoothing,
        max_len=max_len,
        evaluation_frequency=evaluation_frequency
    )
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, config)
    trainer.train(num_iterations)

