import gym
import torch
import torch.nn.functional as f

import critic.advantages
import critic.value_td
import policies.value
import torch_util
import trainers.experience_replay
import trainers.online_trainer
import trainers.synchronous


class QNetwork(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        h1dim = 80
        self._h1 = torch.nn.Linear(num_states, h1dim)
        self._h2 = torch.nn.Linear(h1dim, num_actions)

    def forward(self, states: torch_util.FloatTensor):
        h1 = f.relu(self._h1(states))
        h2 = self._h2(h1)
        return h2


def _run():
    envs = [gym.envs.make("Acrobot-v1") for _ in range(16)]
    num_states = envs[0].observation_space.shape[0]
    num_actions = envs[0].action_space.n
    q_network = QNetwork(num_states, num_actions)
    q_network.cuda()
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=1e-3)
    q_learner = critic.value_td.QValueTD(q_network, 100, gradient_clip=None)
    policy = policies.value.EpsilonGreedy(num_actions, q_network, 1, 0, 0.0001)
    config = trainers.online_trainer.TrainerConfig(
        optimizer=optimizer,
        num_actions=num_actions,
        state_dim=num_states,
        actor=policy,
        critic=q_learner,
        policy=policy,
        advantage_provider=critic.advantages.NoAdvantageProvider(),
        discount_factor=1,
        reward_log_smoothing=0.1
    )
    trainer = trainers.synchronous.SynchronizedDiscreteNstepTrainer(envs, config, 32, 32)
    trainer.train(100000)


if __name__ == '__main__':
    _run()



