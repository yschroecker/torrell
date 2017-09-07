import torch
import tqdm
import numpy as np

from actor.actor_base import DiscreteActor
from environments.typing import Environment
from critic.temporal_difference import TemporalDifference, Batch
import torch_util


class DiscreteOnlineTrainer:
    def __init__(self, env: Environment[int], num_actions: int, actor: DiscreteActor, critic: TemporalDifference,
                 discount_factor: float, batch_size: int=1):
        self._env = env
        self._num_actions = num_actions
        self._actor = actor
        self._critic = critic
        self._discount_factor = discount_factor
        self._batch_size = batch_size

    def _choose_action(self, state: np.ndarray) -> int:
        action_probabilities = self._actor.probabilities(
            torch.autograd.Variable(torch.from_numpy(np.atleast_2d(state)).type(torch_util.Tensor), volatile=True)
        )
        return np.random.choice(self._num_actions, p=action_probabilities.numpy()[0])

    def train(self, num_iterations: int, reward_log_smoothing: float=0.99):
        next_state = self._env.reset()
        next_action = self._choose_action(next_state)
        reward_ema = 0
        episode_reward = 0
        t = 0
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            states = []
            actions = []
            rewards = []
            terminal_states = []
            next_states = []
            next_actions = []
            for _ in range(self._batch_size):
                state = next_state
                action = next_action
                next_state, reward, is_terminal, _ = self._env.step(action)
                episode_reward += self._discount_factor**t * reward
                t += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                terminal_states.append(is_terminal)
                next_states.append(next_state)
                next_actions.append(next_action)

                if is_terminal:
                    reward_ema = (1 - reward_log_smoothing) * reward_ema + reward_log_smoothing * episode_reward
                    trange.set_description(f"r: {reward_ema}, iteration: {iteration}")
                    next_state = self._env.reset()
                    t = 0
                    episode_reward = 0
                next_action = self._choose_action(next_state)
            bootstrap_weights = self._discount_factor * (1 - np.array(terminal_states, dtype=np.float32))
            batch = Batch(
                states=torch.from_numpy(np.array(states, dtype=np.float32)).type(torch_util.Tensor),
                actions=torch.from_numpy(np.array(actions, dtype=np.int32)).type(torch_util.LongTensor),
                intermediate_returns=torch.from_numpy(np.array(rewards, dtype=np.float32)).type(torch_util.Tensor),
                bootstrap_weights=torch.from_numpy(bootstrap_weights).type(torch_util.Tensor),
                bootstrap_states=torch.from_numpy(np.array(next_states, dtype=np.float32)).type(torch_util.Tensor),
                bootstrap_actions=torch.from_numpy(np.array(next_actions, dtype=np.int32)).type(torch_util.LongTensor))
            self._critic.update(batch)
            self._actor.update()
