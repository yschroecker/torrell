import torch

# import environments.breakout
import chainerrl.envs.ale
import networks.simple_shared
import critic.advantages
import critic.value_td
import actor.likelihood_ratio_gradient
import trainers.online_trainer
import trainers.synchronous
import policies.softmax


def _run():
    # envs = [environments.breakout.Breakout('/home/yannick/breakout_monitor') for _ in range(16)]
    envs = [chainerrl.envs.ale.ALE('breakout') for _ in range(16)]
    num_states = [4, 84, 84]
    image_width = 84
    image_height = 84
    history_length = 4
    num_actions = envs[0].action_space.n
    shared_network = networks.simple_shared.SimpleSharedNetwork(
        image_width, image_height, history_length, num_actions
    )
    shared_network.cuda()

    optimizer = torch.optim.RMSprop(shared_network.parameters(), lr=7e-4*16, eps=0.1)
    tdv = critic.value_td.ValueTD(networks.simple_shared.VNetwork(shared_network), target_update_rate=1000)
    softmax_policy = policies.softmax.SoftmaxPolicy(networks.simple_shared.PolicyNetwork(shared_network))
    pg = actor.likelihood_ratio_gradient.LikelihoodRatioGradient(softmax_policy, entropy_regularization=0.01)

    num_samples=15000000
    batch_size=32
    #scheduler = torch.optim.lr_scheduler.LambdaLR(
        #optimizer, lambda iteration: 1-iteration*batch_size/num_samples
    #)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=(num_samples/batch_size)/15, gamma=0.5
    )
    config = trainers.online_trainer.TrainerConfig(
        num_actions=num_actions,
        state_dim=num_states,
        actor=pg,
        critic=tdv,
        policy=softmax_policy,
        optimizer=optimizer,
        advantage_provider=critic.advantages.TDErrorAdvantageProvider(tdv),
        discount_factor=0.99,
        reward_log_smoothing=0.1,
        gradient_clipping=1,
        evaluation_frequency=1000,
        hooks=[
            #(100000, lambda iteration: torch.save(shared_network, 
                #f"/home/yannick/breakout_policies_other_decay/{iteration}")),
            (1, lambda _: scheduler.step())
        ]
    )
    trainer = trainers.synchronous.SynchronizedDiscreteNstepTrainer(envs, config, 5, batch_size)
    trainer.train(num_samples/batch_size)

if __name__ == '__main__':
    _run()

