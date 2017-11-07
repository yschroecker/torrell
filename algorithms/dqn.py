from typing import Optional

import torch

import policies.value
import trainers.experience_replay
import trainers.online_trainer
import optimization_strategies.critic_only
from critic.control.q_learning import DiscreteQLearning, DiscreteDoubleQLearning
from environments.environment import Environment


def train(num_iterations: int, env: Environment[int], q_network: torch.nn.Module, state_dim: int, num_actions: int,
          discount_factor: float, lr: float, target_update_rate: int, memory_size: int, batch_size: int,
          reward_log_smoothing: float, initial_population: int, initial_epsilon: float, epsilon_decay: float,
          final_epsilon: float, gradient_clip: Optional[float]=None, max_len: int=-1,
          evaluation_frequency: int=-1, double_q: bool=False):
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=lr)
    q_learner_type = DiscreteDoubleQLearning if double_q else DiscreteQLearning
    q_learner = q_learner_type(q_network, target_update_rate)
    policy_model = policies.value.ValuePolicyModel(num_actions, q_network)
    strategy = optimization_strategies.critic_only.CriticOnly(optimizer, q_learner, gradient_clip)
    config = trainers.online_trainer.TrainerConfig(
        state_dim=state_dim,
        policy_model=policy_model,
        policy_builder=policies.value.epsilon_greedy(initial_epsilon, final_epsilon, epsilon_decay),
        optimization_strategy=strategy,
        discount_factor=discount_factor,
        reward_log_smoothing=reward_log_smoothing,
        max_len=max_len,
        evaluation_frequency=evaluation_frequency
    )
    trainer = trainers.experience_replay.DiscreteExperienceReplay(
        env, config, memory_size, batch_size, initial_population
    )
    trainer.train(num_iterations)
