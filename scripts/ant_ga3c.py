import torch
import torch.nn.functional as f
import roboschool
import gym

import torch_util
import policies.gaussian
import environments.environment
import algorithms.ga3c


class VNetwork(torch.nn.Module):
    def __init__(self, num_states: int):
        super().__init__()
        hdim = 100
        self._h1 = torch.nn.Linear(num_states, hdim)
        self._h2 = torch.nn.Linear(hdim, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)

    def forward(self, states: torch_util.FloatTensor):
        x = f.relu(self._h1(states))
        x = f.relu(self._h2(x))
        return self._v_out(x)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_states: int, action_dim: int):
        super().__init__()
        hdim = 100
        self._h1p = torch.nn.Linear(num_states, hdim)
        self._h2p = torch.nn.Linear(hdim, hdim)

        self._pi_out = torch.nn.Linear(hdim, action_dim)
        init_logstddev = torch.FloatTensor([-1] * action_dim)
        self._logstddev = torch.nn.Parameter(init_logstddev, requires_grad=True)
        self._action_dim = action_dim

    def forward(self, states: torch_util.FloatTensor):
        x = f.relu(self._h1p(states))
        x = f.relu(self._h2p(x))
        return torch.cat([self._pi_out(x),
                          self._logstddev.expand([states.size(0), self._action_dim])], dim=1)


class EnvWrapper(environments.environment.Environment):
    def __init__(self, env):
        self._env = env

    def step(self, action):
        next_state, reward, is_terminal, info = self._env.step(action)
        return next_state, reward, is_terminal, info

    def reset(self):
        return self._env.reset()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space


def _run():
    envs = [EnvWrapper(gym.make("RoboschoolAnt-v1")) for _ in range(16)]
    env = envs[0]
    # env = gym.wrappers.Monitor(env, "/tmp/", force=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    v_network = VNetwork(state_dim)
    v_network.cuda()
    policy_network = PolicyNetwork(state_dim, action_dim)
    policy_network.cuda()
    policy_model = policies.gaussian.SphericalGaussianPolicyModel(action_dim, policy_network, None)
    algorithms.ga3c.train(
        num_iterations=1000000,
        envs=envs,
        state_dim=state_dim,
        value_network=v_network,
        policy_model=policy_model,
        policy_builder=policies.gaussian.SphericalGaussianPolicy,
        learning_rate=0.001,
        discount_factor=0.99,
        look_ahead=4,
        batch_size=64,
        entropy_regularization=0.01,
        gradient_clipping=10,
        reward_log_smoothing=0.1)


if __name__ == '__main__':
    _run()
