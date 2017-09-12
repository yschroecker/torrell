import algorithms.dqn
import environments.breakout
import networks.simple_conv


def _run():
    env = environments.breakout.Breakout('/home/yannick/breakout_monitor')
    num_states = env.state_dim
    num_actions = env.num_actions
    q_network = networks.simple_conv.SimpleConvQNetwork(env.image_width, env.image_height, env.history_length,
                                                        num_actions)
    q_network.cuda()
    algorithms.dqn.dqn(
        env=env,
        q_network=q_network,
        state_dim=num_states,
        num_actions=num_actions,
        discount_factor=0.99,
        lr=0.00025,
        num_iterations=10000000,
        target_update_rate=10000,
        memory_size=100000,
        batch_size=32,
        reward_log_smoothing=0.1,
        initial_population=10000,
        initial_epsilon=1.,
        epsilon_decay=1e-5,
        final_epsilon=0.1
    )


if __name__ == '__main__':
    _run()
