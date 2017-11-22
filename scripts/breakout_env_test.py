import chainerrl.envs.ale
import scipy.misc


def _run():
    env = chainerrl.envs.ale.ALE('breakout')
    t = 0
    while True:
        env.reset()
        is_terminal = False
        while not is_terminal:
            if t % 1000 == 0:
                print(t)
                scipy.misc.imsave(f"breakout_images/{t}.png", env.ale.getScreenRGB())
            t += 1
            _, _, is_terminal, _ = env.step(3)
            if t % 2000 == 0:
                env.ale.reset_game()
                env.reset()


if __name__ == '__main__':
    _run()
