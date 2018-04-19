from typing import Optional

import torch

import policies.value
import trainers.experience_replay
import trainers.data_collection
import algorithms.algorithm
import core_algorithms.critic_only
from critic.control.q_learning import DiscreteQLearning, DiscreteDoubleQLearning
from environments.environment import Environment


class DQN(algorithms.algorithm.Algorithm):
    def __init__(self, env: Environment[int], q_network: torch.nn.Module, state_dim: int, num_actions: int,
                 discount_factor: float, lr: float, target_update_rate: int, memory_size: int, batch_size: int,
                 reward_log_smoothing: float, initial_population: int, initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, gradient_clip: Optional[float]=None, max_len: int=-1, double_q: bool=False):
        optimizer = torch.optim.RMSprop(q_network.parameters(), lr=lr)
        q_learner_type = DiscreteDoubleQLearning if double_q else DiscreteQLearning
        q_learner = q_learner_type(q_network, target_update_rate, gradient_clip=gradient_clip)
        policy_model = policies.value.ValuePolicyModel(num_actions, q_network)
        policy_builder = policies.value.epsilon_greedy(initial_epsilon, final_epsilon, epsilon_decay)
        self.policy = policy_builder(policy_model)
        self._trainer_config = trainers.data_collection.TrainerConfig(
            state_dim=state_dim,
            action_dim=num_actions,
            discount_factor=discount_factor,
            max_len=max_len,
            reward_log_smoothing=reward_log_smoothing
        )
        self._trainer = trainers.data_collection.Trainer(env, self.policy, self._trainer_config)
        self._memory = trainers.experience_replay.MixedBatchExperienceReplay(self._trainer, memory_size, batch_size,
                                                                             initial_population)
        self._core = core_algorithms.critic_only.CriticOnly(optimizer, q_learner, gradient_clip)
        super().__init__([self._trainer], self._trainer_config, self.policy)

    def iterate(self, iteration: int):
        batch = self._memory.collect_transitions()
        if batch is not None:
            self._core.iterate(iteration, batch)

