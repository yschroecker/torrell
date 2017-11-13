import abc

import torch
import torch.nn.functional as f

import torch_util
import environments.cartpole
import algorithms.discrete_a2c


class SharedNetwork(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        hdim = 200
        self._h1 = torch.nn.Linear(num_states, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)
        self._pi_out = torch.nn.Linear(hdim, num_actions)

    def shared(self, states: torch_util.FloatTensor):
        h1 = f.relu(self._h1(states))
        return h1

    def v(self, states: torch_util.FloatTensor):
        return self._v_out(self.shared(states))

    def pi(self, states: torch_util.FloatTensor):
        return self._pi_out(self.shared(states))


class VNetwork(torch.nn.Module):
    def __init__(self, shared: torch.nn.Module):
        super().__init__()
        self._shared = shared

    def forward(self, states: torch_util.FloatTensor):
        return self._shared.v(states)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, shared: torch.nn.Module):
        super().__init__()
        self._shared = shared

    def forward(self, states: torch_util.FloatTensor):
        return self._shared.pi(states)


def _run():
    env = environments.cartpole.Cartpole()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    shared_network = SharedNetwork(num_states, num_actions)
    v_network = VNetwork(shared_network)
    policy_network = PolicyNetwork(shared_network)
    algorithms.discrete_a2c.train(
        num_iterations=10000,
        env=env,
        state_dim=num_states,
        value_network=v_network,
        policy_network=policy_network,
        learning_rate=0.001,
        discount_factor=0.99,
        batch_size=32,
        reward_log_smoothing=0.1,
        max_len=200
    )


if __name__ == '__main__':
    _run()

