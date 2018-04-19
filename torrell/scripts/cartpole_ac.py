import abc

import torch
import torch.nn.functional as f

import torch_util
import environments.cartpole
from torrell import policies
import algorithms.a2c


class VNetwork(torch.nn.Module):
    def __init__(self, num_states: int):
        super().__init__()
        hdim = 80
        self._h1 = torch.nn.Linear(num_states, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)

    def forward(self, states: torch.autograd.Variable):
        h1 = f.relu(self._h1(states))
        return self._v_out(h1)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        hdim = 80
        self._h1 = torch.nn.Linear(num_states, hdim)

        self._pi_out = torch.nn.Linear(hdim, num_actions)

    def forward(self, states: torch.autograd.Variable):
        h1 = f.relu(self._h1(states))
        return self._pi_out(h1)


def _run():
    envs = [environments.cartpole.Cartpole()]
    num_states = envs[0].observation_space.shape[0]
    num_actions = envs[0].action_space.n
    v_network = VNetwork(num_states)
    policy_network = PolicyNetwork(num_states, num_actions)
    policy_model = policies.softmax.SoftmaxPolicyModel(policy_network)
    a2c = algorithms.a2c.A2C(
        envs=envs,
        state_dim=num_states,
        action_dim=num_actions,
        value_network=v_network,
        policy_model=policy_model,
        policy_builder=policies.softmax.SoftmaxPolicy,
        learning_rate=0.01,
        discount_factor=0.9,
        batch_size=32,
        reward_log_smoothing=0.1,
        max_len=200
    )
    for iteration in a2c.rl_eval_range(0, 100000, environments.cartpole.Cartpole(), 100):
        a2c.iterate(iteration)


if __name__ == '__main__':
    _run()

