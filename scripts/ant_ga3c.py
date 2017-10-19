import torch
import torch.nn.functional as f
import numpy as np
import roboschool
import gym

import torch_util
import trainers.online_trainer
import trainers.synchronous
import policies.gaussian
import critic.value_td
import critic.advantages
import actor.likelihood_ratio_gradient


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


class EnvWrapper:
    def __init__(self, env):
        self._env = env

    def step(self, action):
        action = np.clip(action, -10, 10)
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
    num_states = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    v_network = VNetwork(num_states)
    policy_network = PolicyNetwork(num_states, action_dim)
    optimizer = torch.optim.RMSprop(list(v_network.parameters()) + list(policy_network.parameters()), lr=0.0001)
    tdv = critic.value_td.ValueTD(v_network, target_update_rate=1)
    policy = policies.gaussian.SphericalGaussianPolicy(action_dim, policy_network, None, noise_sample_rate=1)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(policy, entropy_regularization=0)
    config = trainers.online_trainer.TrainerConfig(
        action_type=np.float32,
        state_dim=num_states,
        actor=pg,
        critic=tdv,
        policy=policy,
        optimizer=optimizer,
        advantage_provider=critic.advantages.TDErrorAdvantageProvider(tdv),
        discount_factor=0.99,
        reward_log_smoothing=0.1
    )
    trainer = trainers.synchronous.SynchronizedDiscreteNstepTrainer(envs, config, 10, 64)
    # trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, config, batch_size=32)
    trainer.train(100000)


if __name__ == '__main__':
    _run()
