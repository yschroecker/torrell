from typing import Optional

import torch

from environments.environment import Environment
import critic.control.q_learning
import critic.advantages
import actor.value
import trainers.experience_replay
import trainers.online_trainer


def dqn(env: Environment[int], q_network: torch.nn.Module, state_dim: int, num_actions: int, discount_factor: float,
        lr: float, num_iterations: int, target_update_rate: int, memory_size: int, batch_size: int,
        reward_log_smoothing: float, initial_population: int, initial_epsilon: float, epsilon_decay: float,
        final_epsilon: float, gradient_clip: Optional[float]=None, maxlen: int=-1) -> actor.value.EpsilonGreedy:
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=lr)
    q_learner = critic.control.q_learning.DiscreteQLearning(q_network, optimizer, target_update_rate,
                                                            gradient_clip=gradient_clip)
    policy = actor.value.EpsilonGreedy(num_actions, q_network, initial_epsilon, final_epsilon, epsilon_decay)
    config = trainers.online_trainer.TrainerConfig(
        env=env,
        num_actions=num_actions,
        state_dim=state_dim,
        actor=policy,
        critic=q_learner,
        policy=policy,
        advantage_provider=critic.advantages.NoAdvantageProvider(),
        discount_factor=discount_factor,
        reward_log_smoothing=reward_log_smoothing,
        maxlen=maxlen
    )
    trainer = trainers.experience_replay.DiscreteExperienceReplay(config, memory_size, batch_size, initial_population)
    trainer.train(num_iterations)
    return policy
