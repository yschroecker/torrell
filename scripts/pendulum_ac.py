import abc

import torch
import torch.nn.functional as f
import numpy as np
import gym

import torch_util
import trainers.online_trainer
import environments.cartpole
import policies.gaussian
import critic.value_td
import critic.advantages
import actor.likelihood_ratio_gradient


class VNetwork(torch.nn.Module):
    def __init__(self, num_states: int):
        super().__init__()
        hdim = 100
        self._h1 = torch.nn.Linear(num_states, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)

    def forward(self, states: torch_util.FloatTensor):
        x = f.tanh(self._h1(states))
        return self._v_out(x)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_states: int, action_dim: int):
        super().__init__()
        hdim = 100
        self._h1p = torch.nn.Linear(num_states, hdim)

        self._pi_out = torch.nn.Linear(hdim, action_dim)
        init_logstddev = torch.FloatTensor([-5])
        self._logstddev = torch.nn.Parameter(init_logstddev, requires_grad=True)
        self._action_dim = action_dim

    def forward(self, states: torch_util.FloatTensor):
        x = f.tanh(self._h1p(states))
        return torch.cat([self._pi_out(x),
                          self._logstddev.expand([states.size(0), self._action_dim])], dim=1)


def _run():
    env = gym.make("Pendulum-v0")
    num_states = env.observation_space.shape[0]
    action_dim = 1
    v_network = VNetwork(num_states)
    policy_network = PolicyNetwork(num_states, action_dim)
    optimizer = torch.optim.RMSprop(list(v_network.parameters()) + list(policy_network.parameters()), lr=0.1)
    tdv = critic.value_td.ValueTD(v_network, target_update_rate=1)
    policy = policies.gaussian.SphericalGaussianPolicy(action_dim, policy_network, None)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(policy, entropy_regularization=0)
    config = trainers.online_trainer.TrainerConfig(
        action_type=np.float32,
        state_dim=num_states,
        actor=pg,
        critic=tdv,
        policy=policy,
        optimizer=optimizer,
        advantage_provider=critic.advantages.TDErrorAdvantageProvider(tdv),
        discount_factor=0.5,
        reward_log_smoothing=0.1,
    )
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, config, batch_size=32)
    trainer.train(10000)


if __name__ == '__main__':
    _run()
