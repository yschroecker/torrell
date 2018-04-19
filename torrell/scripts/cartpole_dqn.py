import environments.cartpole
import torch
import torch.nn.functional as f
import torch_util

import torrell.algorithms.dqn


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
    env = environments.cartpole.Cartpole()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    q_network = QNetwork(num_states, num_actions)
    q_network.cuda()
    dqn = torrell.algorithms.dqn.DQN(
        env=env,
        q_network=q_network,
        state_dim=num_states,
        num_actions=num_actions,
        discount_factor=1,
        lr=1e-3,
        target_update_rate=100,
        memory_size=100000,
        batch_size=32,
        reward_log_smoothing=0.1,
        initial_population=320,
        initial_epsilon=0,
        epsilon_decay=0,
        final_epsilon=0,
        max_len=200,
        double_q=True
    )
    for iteration in dqn.rl_eval_range(0, 100000):
        dqn.iterate(iteration)


if __name__ == '__main__':
    _run()
