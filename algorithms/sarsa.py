import torch

import critic.advantages
import critic.value_td
import environments.environment
import policies.value
import trainers.data_collection
import core_algorithms.critic_only
import algorithms.algorithm


class SARSA(algorithms.algorithm.Algorithm):
    def __init__(self, env: environments.environment.Environment,  state_dim: int, num_actions: int,
                 q_network: torch.nn.Module, learning_rate: float, discount_factor: float,
                 initial_epsilon: float=0, final_epsilon: float=0, epsilon_decay_rate: float=1,
                 reward_log_smoothing: float=1, max_len: int=-1):
        q_learner = critic.value_td.QValueTD(q_network, 1)
        epsilon_greedy = policies.value.ValuePolicyModel(num_actions, q_network)
        policy_builder = policies.value.epsilon_greedy(initial_epsilon, final_epsilon, epsilon_decay_rate)
        self.policy = policy_builder(epsilon_greedy)
        optimizer = torch.optim.SGD(q_network.parameters(), lr=learning_rate)
        self._trainer_config = trainers.data_collection.TrainerConfig(
            state_dim=state_dim,
            action_dim=num_actions,
            discount_factor=discount_factor,
            max_len=max_len,
            reward_log_smoothing=reward_log_smoothing
        )
        self._trainer = trainers.data_collection.Trainer(env, self.policy, self._trainer_config)
        self._core = core_algorithms.critic_only.CriticOnly(optimizer, q_learner)
        super().__init__([self._trainer], self._trainer_config, self.policy)

    def iterate(self, iteration: int):
        batch = trainers.data_collection.collect_batch(self._trainer, 1, 1)
        self._core.iterate(iteration, batch)

