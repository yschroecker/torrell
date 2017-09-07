import torch
import torch.nn.functional as f

import trainers.online_trainer
import environments.cartpole
import actor.value
import critic.control.q_learning
import torch_util


class QNetwork(torch.nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        h1dim = 80
        self._h1 = torch.nn.Linear(num_states, h1dim)
        self._h2 = torch.nn.Linear(h1dim, num_actions)

    def forward(self, states: torch_util.Tensor):
        h1 = f.relu(self._h1(states))
        h2 = self._h2(h1)
        return h2


def _run():
    env = environments.cartpole.Cartpole()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    q_network = QNetwork(num_states, num_actions)
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=1e-3)
    q_learner = critic.control.q_learning.DiscreteQLearning(q_network, optimizer, 100)
    policy = actor.value.greedy(num_actions, q_network)
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, num_actions, policy, q_learner, discount_factor=1,
                                                            batch_size=32, reward_log_smoothing=0.01, maxlen=200)
    trainer.train(10000)


if __name__ == '__main__':
    _run()
