import torch

# import environments.breakout
import numpy as np
import scipy.misc
import random

import chainerrl.envs.ale
import networks.simple_shared
import critic.advantages
import critic.value_td
import critic.retrace
import actor.likelihood_ratio_gradient
import trainers.online_trainer
import trainers.synchronous
import policies.softmax
import visualization
import core_algorithms.actor_critic


def _run():
    # envs = [environments.breakout.Breakout('/home/yannick/breakout_monitor') for _ in range(16)]
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(2)
    torch.cuda.manual_seed(3)
    envs = [chainerrl.envs.ale.ALE('breakout', seed=i) for i in range(16)]
    num_states = [4, 84, 84]
    image_width = 84
    image_height = 84
    history_length = 4
    num_actions = envs[0].action_space.n
    shared_network = networks.simple_shared.SimpleSharedNetwork(
        image_width, image_height, history_length, num_actions
    )
    shared_network.cuda()

    optimizer = torch.optim.RMSprop(shared_network.parameters(), lr=7e-4, eps=0.1)
    softmax_policy = policies.softmax.SoftmaxPolicyModel(networks.simple_shared.PolicyNetwork(shared_network))
    tdv = critic.value_td.ValueTD(networks.simple_shared.VNetwork(shared_network), target_update_rate=1)
    # tdv = critic.retrace.Retrace(networks.simple_shared.VNetwork(shared_network), softmax_policy, softmax_policy,
                                 # lambda_decay=1)
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(softmax_policy, entropy_regularization=0.01)

    num_samples = 30000000
    batch_size = 32
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    # optimizer, lambda iteration: 1 - iteration * batch_size / num_samples
    # )

    # scheduler = torch.optim.lr_scheduler.StepLR(
    # optimizer, step_size=(num_samples/batch_size)/6, gamma=0.5
    # )
    def image_summary(iteration):
        idx = np.random.choice(16)
        if iteration > 0:
            scipy.misc.imsave(f"breakout_images_preprocessed/{iteration}.png", envs[idx].state[-1])
        image = envs[idx].ale.getScreenRGB()
        visualization.reporting.global_summary_writer.add_image(f'state_{idx}', image, iteration)

    strategy = core_algorithms.actor_critic.ActorCritic(
        optimizer, pg, tdv, critic.advantages.TDErrorAdvantageProvider(tdv), gradient_clipping=1,
    )

    config = trainers.online_trainer.TrainerConfig(
        state_dim=num_states,
        optimization_strategy=strategy,
        policy_model=softmax_policy,
        policy_builder=policies.softmax.SoftmaxPolicy,
        reward_clipping=[-1, 1],
        discount_factor=0.99,
        reward_log_smoothing=0.1,
        evaluation_frequency=50,
        max_len=10000,
        hooks=[
            (100000, lambda iteration: torch.save(shared_network,
                                                  f"/home/yannick/breakout_policies/{iteration}")),
            (10, image_summary)
            # (1, lambda _: scheduler.step())
        ]
    )
    # noinspection PyTypeChecker
    trainer = trainers.synchronous.SynchronizedDiscreteNstepTrainer(envs, config, 4, batch_size)
    trainer.train(num_samples // batch_size)


if __name__ == '__main__':
    _run()
