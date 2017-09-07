import torch

from environments.typing import Environment
import critic.control.q_learning
import actor.value
import trainers.experience_replay


def dqn(env: Environment[int], q_network: torch.nn.Module, state_dim: int, num_actions: int, discount_factor: float,
        lr: float, num_iterations: int, target_update_rate: int, memory_size: int, batch_size: int,
        reward_log_smoothing: float, initial_population: int, initial_epsilon: float, epsilon_decay: float,
        final_epsilon: float, maxlen: int=-1) -> actor.value.EpsilonGreedy:
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=lr)
    q_learner = critic.control.q_learning.DiscreteQLearning(q_network, optimizer, target_update_rate)
    policy = actor.value.EpsilonGreedy(num_actions, q_network, initial_epsilon, epsilon_decay, final_epsilon)
    trainer = trainers.experience_replay.DiscreteExperienceReplay(
        env, state_dim, num_actions, policy, q_learner, discount_factor, reward_log_smoothing, memory_size, batch_size,
        initial_population, maxlen)
    trainer.train(num_iterations)
    return policy
