from OpenGL import GLU
import torch
import torch.nn.functional as f
import roboschool
import gym

import torch_util
import critic.advantages
from torrell import policies
import environments.environment
import algorithms.a2c


class VNetwork(torch.nn.Module):
    def __init__(self, num_states: int):
        super().__init__()
        hdim = 100
        self._h1 = torch.nn.Linear(num_states, hdim)
        self._h2 = torch.nn.Linear(hdim, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)

    def forward(self, states: torch_util.FloatTensor):
        x = f.tanh(self._h1(states))
        x = f.tanh(self._h2(x))
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
        x = f.tanh(self._h1p(states))
        x = f.tanh(self._h2p(x))
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


def show_run(policy: policies.gaussian.SphericalGaussianPolicy):
    env = gym.make('RoboschoolAnt-v1')
    while True:
        state = env.reset()
        score = 0
        done = False
        t = 0
        state100 = None
        while not done:
            t += 1
            if t == 100 and state100 is None:
                state100 = state
                t = 0
            elif t == 100:
                for i, joint in enumerate(env.env.ordered_joints):
                    # joint.reset_current_position(float(state100[i*2]), float(state100[i*2+1]))
                    # pos, vel = joint.current_relative_position()
                    pos, vel = float(state100[2*i + 8]), float(state100[2*i + 9])
                    joint_limit1, joint_limit2 = joint.limits()[:2]
                    pos_mid = 0.5 * (joint_limit2 + joint_limit1)
                    print(pos, vel)
                    new_pos = (pos * (joint_limit2 - joint_limit1)/2) + pos_mid
                    new_vel = vel*10
                    new_pos = (pos * (joint_limit2 - joint_limit1)/2) + pos_mid
                    joint.reset_current_position(new_pos, new_vel)
                    print((new_pos, new_vel))
                t = 0
            action = policy.sample(state, 0)
            state, r, done, _ = env.step(action)
            score += r
            still_open = env.render('human')
            # if t == 0:
            #     print("<<<<<<<<<<<")
            #     for i, joint in enumerate(env.env.ordered_joints):
            #         # joint.reset_current_position(float(state100[i*2]), float(state100[i*2+1]))
            #         print(joint.current_relative_position())
            #     print(">>>>>>>>>>>")
        print("score=%0.2f" % score)



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

    a2c = algorithms.a2c.A2C(
        envs=envs,
        state_dim=state_dim,
        action_dim=action_dim,
        value_network=v_network,
        policy_model=policy_model,
        policy_builder=policies.gaussian.SphericalGaussianPolicy,
        learning_rate=0.001,
        discount_factor=0.99,
        steps_per_agent=4,
        batch_size=64,
        entropy_regularization=0.01,
        gradient_clipping=10,
        reward_log_smoothing=0.1,
        lr_decay=0.99997,
        advantage_type=critic.advantages.TDErrorAdvantageProvider,
        # advantage_provider_args={'mean_reward_lr': 1e-6}
    )
    for iteration in a2c.rl_eval_range(0, 1000000, EnvWrapper(gym.make("RoboschoolAnt-v1")), 100):
        a2c.iterate(iteration)

        if iteration % 1000 == 0:
            torch.save(policy_model, f"policies/trained_ant_policy2_{iteration}")

    # torch.save(policy_model, "trained_ant_policy")
    # show_run(policies.gaussian.SphericalGaussianPolicy(policy_model))


if __name__ == '__main__':
    policy_model = torch.load("policies/trained_ant_policy_565000")
    show_run(policies.gaussian.SphericalGaussianPolicy(policy_model))
    # _run()
