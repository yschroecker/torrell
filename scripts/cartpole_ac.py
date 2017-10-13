import torch
import torch.nn.functional as f

import torch_util
import trainers.online_trainer
import environments.cartpole
import policies.softmax
import critic.value_td
import critic.advantages
import actor.likelihood_ratio_gradient

class SharedNetwork(torch.nn.Module):
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

'''
class VNetwork(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        h1dim = 80
        self._h1 = torch.nn.Linear(num_states, h1dim)
        self._h2 = torch.nn.Linear(h1dim, 1)

    def forward(self, states: torch_util.FloatTensor):
        h1 = f.relu(self._h1(states))
        h2 = self._h2(h1)
        return h2

class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        self._linear = torch.nn.Linear(num_states, num_actions)

    def forward(self, states: torch_util.FloatTensor):
        return self._linear(states)
'''


def _run():
    env = environments.cartpole.Cartpole()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    shared_network = SharedNetwork(num_states, num_actions)
    v_network = VNetwork(shared_network)
    policy_network = PolicyNetwork(shared_network)
    optimizer = torch.optim.RMSprop(shared_network.parameters(), lr=0.001)
    tdv = critic.value_td.ValueTD(v_network, target_update_rate=1)
    softmax_policy = policies.softmax.SoftmaxPolicy(policy_network)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(softmax_policy, entropy_regularization=0)
    config = trainers.online_trainer.TrainerConfig(
        num_actions=num_actions,
        state_dim=num_states,
        actor=pg,
        critic=tdv,
        policy=softmax_policy,
        optimizer=optimizer,
        advantage_provider=critic.advantages.TDErrorAdvantageProvider(tdv),
        discount_factor=1,
        reward_log_smoothing=0.1,
        maxlen=200
    )
    trainer = trainers.online_trainer.DiscreteOnlineTrainer(env, config, batch_size=32)
    trainer.train(10000)


if __name__ == '__main__':
    _run()

