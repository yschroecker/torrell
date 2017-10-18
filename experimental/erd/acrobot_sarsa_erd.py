import experimental.erd.erd_trainer
import critic.advantages
import actor.value
import trainers.experience_replay
import trainers.online_trainer
import torch
import torch.nn.functional as f
import critic.value_td
import torch_util
import gym


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


class DiscriminatorNetwork(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        h1dim = 80
        self._h1 = torch.nn.Linear(num_states + num_actions, h1dim)
        self._h2 = torch.nn.Linear(h1dim, 1)

    def forward(self, states: torch_util.FloatTensor):
        h1 = f.relu(self._h1(states))
        h2 = f.logsigmoid(self._h2(h1))
        return h2


def _run():
    env = gym.envs.make("Acrobot-v1")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    q_network = QNetwork(num_states, num_actions)
    q_network.cuda()
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=1e-3)
    q_learner = critic.value_td.QValueTD(q_network, 100, gradient_clip=None)
    policy = actor.value.EpsilonGreedy(num_actions, q_network, 1, 0, 0.0001)
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
    discriminator = DiscriminatorNetwork(num_states, num_actions)
    discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=1e-3)
    trainer = experimental.erd.erd_trainer.DiscreteExperienceReplayWithDiscriminator(
        discriminator, discriminator_optimizer, 960, 100, config, 32000, 32, 1000, 32
    )
    trainer.train(100000)


if __name__ == '__main__':
    _run()


