import gym.envs.classic_control


class Cartpole(gym.envs.classic_control.cartpole.CartPoleEnv):
    def __init__(self):
        super().__init__()

    def step(self, action):
        state, reward, is_terminal, info = super().step(action)
        if is_terminal:
            reward = -100
        return state, reward, is_terminal, info
