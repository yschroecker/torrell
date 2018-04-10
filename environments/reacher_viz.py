from OpenGL import GLU
import roboschool
import gym
import numpy as np

import policies.gaussian


def show_from_state(viz_states, policy: policies.gaussian.SphericalGaussianPolicy, break_every: int=0):
    env = gym.make('RoboschoolReacher-v1')
    for i, viz_state in enumerate(viz_states):
        if break_every > 0 and i % break_every == 0:
            input("Press Enter to continue...")
        state = env.reset()
        score = 0
        done = False
        for i, joint in enumerate(env.env.ordered_joints):
            joint_limit1, joint_limit2 = joint.limits()[:2]
            pos_mid = 0.5 * (joint_limit2 + joint_limit1)
            if joint.name == 'joint0':
                new_pos = np.arctan2(viz_state[5], viz_state[4])
                new_vel = viz_state[6]
                new_vel = new_vel*10
                # new_pos = (pos * (joint_limit2 - joint_limit1)/2) + pos_mid
            elif joint.name == 'joint1':
                pos = viz_state[7]
                vel = viz_state[8]
                new_vel = vel*10
                new_pos = (pos * (joint_limit2 - joint_limit1)/2) + pos_mid
            elif joint.name == 'target_x':
                new_pos = viz_state[0]
                new_vel = 0
            elif joint.name == 'target_y':
                new_pos = viz_state[1]
                new_vel = 0
            joint.reset_current_position(float(new_pos), float(new_vel))
        while not done:
            if policy is None:
                action = np.zeros((env.action_space.shape[0],))
            else:
                action = policy.sample(state, 0)
            state, r, done, _ = env.step(action)
            score += r
            still_open = env.render('human')
            # if t == 0:
            #     print("<<<<<<<<<<<")
            #     for i, joint in enumerate(env.env.ordered_joints):
            #         # joint.reset_current_position(float(state10[i*2]), float(state10[i*2+1]))
            #         print(joint.current_relative_position())
                #     print(">>>>>>>>>>>")
        print("score=%0.2f" % score)

