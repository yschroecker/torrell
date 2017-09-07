import torch
import tqdm
import numpy as np

from actor.actor_base import DiscreteActor
from environments.typing import Environment
from critic.temporal_difference import TemporalDifference, Batch
import torch_util


class DiscreteTrainer:
    def __init__(self, env: Environment[int], num_actions: int, actor: DiscreteActor, critic: TemporalDifference,
                 discount_factor: float, reward_log_smoothing: float, maxlen: int=-1):
        self._env = env
        self._num_actions = num_actions
        self._actor = actor
        self._critic = critic
        self._discount_factor = discount_factor

        self._reward_log_smoothing = reward_log_smoothing
        self._maxlen = maxlen
        self._t = 0
        self._reward_ema = 0
        self._episode_reward = 0
        self._next_state = env.reset()
        self._next_action = self._choose_action(self._next_state)
        self._episode = 0

    def _choose_action(self, state: np.ndarray) -> int:
        action_probabilities = self._actor.probabilities(
            torch.autograd.Variable(torch.from_numpy(np.atleast_2d(state)).type(torch_util.Tensor), volatile=True)
        )
        return np.random.choice(self._num_actions, p=action_probabilities.numpy()[0])

    def collect_transitions(self, num_steps: int):
        states = []
        actions = []
        rewards = []
        terminal_states = []
        next_states = []
        next_actions = []
        for _ in range(num_steps):
            state = self._next_state
            action = self._next_action
            self._next_state, reward, is_terminal, _ = self._env.step(action)
            self._episode_reward += self._discount_factor**self._t * reward
            self._t += 1

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminal_states.append(is_terminal)
            next_states.append(self._next_state)
            next_actions.append(self._next_action)

            if is_terminal or self._t == self._maxlen:
                torch_util.global_summary_writer.add_scalar('episode reward', self._episode_reward, self._episode)
                self._reward_ema = (1 - self._reward_log_smoothing) * self._reward_ema + \
                                   self._reward_log_smoothing * self._episode_reward
                self._next_state = self._env.reset()
                self._t = 0
                self._episode_reward = 0
                self._episode += 1
            self._next_action = self._choose_action(self._next_state)
        return states, actions, rewards, terminal_states, next_states, next_actions

    def _train(self, iteration: int, batch: Batch) -> str:
        self._critic.update(batch)
        self._actor.update()
        return f"r: {self._reward_ema}, iteration: {iteration}"


class DiscreteOnlineTrainer(DiscreteTrainer):
    def __init__(self, env: Environment[int], num_actions: int, actor: DiscreteActor, critic: TemporalDifference,
                 discount_factor: float, reward_log_smoothing: float, maxlen: int=-1, batch_size: int=1):
        self._batch_size = batch_size
        super().__init__(env, num_actions, actor, critic, discount_factor, reward_log_smoothing, maxlen)

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            states, actions, rewards, terminal_states, next_states, next_actions = \
                self.collect_transitions(self._batch_size)

            bootstrap_weights = self._discount_factor * (1 - np.array(terminal_states, dtype=np.float32))
            # noinspection PyUnresolvedReferences
            batch = Batch.from_numpy(
                states=np.array(states, dtype=np.float32),
                actions=np.array(actions, dtype=np.int32),
                intermediate_returns=np.array(rewards, dtype=np.float32),
                bootstrap_weights=bootstrap_weights,
                bootstrap_states=np.array(next_states, dtype=np.float32),
                bootstrap_actions=np.array(next_actions, dtype=np.int32)
            )
            trange.set_description(self._train(iteration, batch))


class DiscreteNstepTrainer(DiscreteTrainer):
    def __init__(self, env: Environment[int], num_actions: int, actor: DiscreteActor, critic: TemporalDifference,
                 discount_factor: float, reward_log_smoothing: float, maxlen: int=-1, batch_size: int=1):
        self._batch_size = batch_size
        super().__init__(env, num_actions, actor, critic, discount_factor, reward_log_smoothing, maxlen)

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            states, actions, rewards, terminal_states, next_states, next_actions = \
                self.collect_transitions(self._batch_size)

            final_state = next_states[-1]
            final_action = next_actions[-1]
            bootstrap_states = [final_state]
            bootstrap_actions = [final_action]
            bootstrap_weights = [self._discount_factor * (1 - terminal_states[-1])]
            current_return = rewards[-1]
            intermediate_returns = [current_return]
            for i in range(len(states) - 2, -1, -1):
                if terminal_states[i]:
                    final_state = next_states[i]
                    final_action = next_actions[i]
                    bootstrap_weights.append(0.)
                    current_return = rewards[i]
                else:
                    bootstrap_weights.append(bootstrap_weights[-1] * self._discount_factor)
                    current_return = rewards[i] + self._discount_factor*current_return
                intermediate_returns.append(current_return)
                bootstrap_states.append(final_state)
                bootstrap_actions.append(final_action)

            bootstrap_weights = np.array(bootstrap_weights[::-1], dtype=np.float32)
            bootstrap_states = bootstrap_states[::-1]
            bootstrap_actions = bootstrap_actions[::-1]
            intermediate_returns = intermediate_returns[::-1]

            # noinspection PyUnresolvedReferences
            batch = Batch.from_numpy(
                states=np.array(states, dtype=np.float32),
                actions=np.array(actions, dtype=np.int32),
                intermediate_returns=np.array(intermediate_returns, dtype=np.float32),
                bootstrap_weights=np.array(bootstrap_weights, dtype=np.float32),
                bootstrap_states=np.array(bootstrap_states, dtype=np.float32),
                bootstrap_actions=np.array(bootstrap_actions, dtype=np.int32)
            )
            trange.set_description(self._train(iteration, batch))

