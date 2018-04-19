import actor.likelihood_ratio_gradient
import critic.advantages
import critic.value_td
import gym
import policies.softmax
import torch
import torch.nn.functional as f
import torch_util
import trainers.online_trainer
import trainers.synchronous

import torrell.core_algorithms.actor_critic


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        h1dim = 80
        self._h1 = torch.nn.Linear(num_states, h1dim)
        self._h2 = torch.nn.Linear(h1dim, num_actions)

    def forward(self, states: torch_util.FloatTensor):
        h1 = f.relu(self._h1(states))
        h2 = self._h2(h1)
        return h2


class VNetwork(torch.nn.Module):
    def __init__(self, num_states: int):
        super().__init__()
        h1dim = 80
        self._h1 = torch.nn.Linear(num_states, h1dim)
        self._h2 = torch.nn.Linear(h1dim, 1)

    def forward(self, states: torch_util.FloatTensor):
        h1 = f.relu(self._h1(states))
        h2 = self._h2(h1)
        return h2


def _run():
    envs = [gym.envs.make("Acrobot-v1") for _ in range(16)]
    num_states = envs[0].observation_space.shape[0]
    num_actions = envs[0].action_space.n
    policy_network = PolicyNetwork(num_states, num_actions)
    policy_network.cuda()
    value_network = VNetwork(num_states)
    value_network.cuda()

    optimizer = torch.optim.RMSprop(set(policy_network.parameters()) | set(value_network.parameters()), lr=7e-4,
                                    eps=0.1)
    softmax_policy = policies.softmax.SoftmaxPolicyModel(policy_network)
    tdv = critic.value_td.ValueTD(value_network, target_update_rate=1)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(softmax_policy, entropy_regularization=0.01)

    strategy = torrell.core_algorithms.actor_critic.ActorCritic(
        optimizer, pg, tdv, critic.advantages.TDErrorAdvantageProvider(tdv), gradient_clipping=1,
    )

    config = trainers.online_trainer.TrainerConfig(
        state_dim=num_states,
        optimization_strategy=strategy,
        policy_model=softmax_policy,
        policy_builder=policies.softmax.SoftmaxPolicy,
        discount_factor=1,
        reward_log_smoothing=0.25,
        evaluation_frequency=50,
        max_len=10000,
        hooks=[]
    )
    # noinspection PyTypeChecker
    trainer = trainers.synchronous.SynchronizedDiscreteNstepTrainer(envs, config, 4, 32)
    trainer.train(10000)
    torch.save(softmax_policy, "trained_acrobot_policy")


if __name__ == '__main__':
    _run()

