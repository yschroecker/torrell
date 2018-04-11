from typing import Any, Type, Tuple, NamedTuple, Sequence, Callable, Generic, Optional
import abc

import torch.optim
import torch.optim.lr_scheduler
import tqdm
import numpy as np

from actor.actor_base import Actor
from environments.environment import Environment, ActionT
from critic.temporal_difference import TemporalDifferenceBase, Batch
from core_algorithms.optimization_strategy import OptimizationStrategy
from policies.policy import PolicyModel, Policy
import visualization
import data
import tqdm


class TrainerConfig(NamedTuple):
    state_dim: int
    discount_factor: float
    reward_log_smoothing: float = 1
    action_dim: int = 1
    max_len: int = -1


class Trainer(Generic[ActionT]):
    def __init__(self, env: Environment[ActionT], policy: Policy, trainer_config: TrainerConfig,
                 evaluation_mode: bool=False, collect_history: bool=False):
        self._evaluation_mode = evaluation_mode
        self._env = env
        self._policy = policy
        self.discount_factor = trainer_config.discount_factor

        self._reward_log_smoothing = trainer_config.reward_log_smoothing
        self._maxlen = trainer_config.max_len

        self._next_state = env.reset()
        self._next_action = self._choose_action(self._next_state, 0)
        self.reward_ema = 0
        self.score_ema = 0
        self._t = 0
        self._episode = 0
        self.sample_count = 0
        self._episode_reward = 0.
        self._episode_score = 0.

        self._collect_history = collect_history
        if collect_history:
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []

    def _choose_action(self, state: np.ndarray, t: int) -> ActionT:
        return self._policy.sample(state, t, not self._evaluation_mode)

    def collect_sequence(self, num_steps: int=-1) -> data.RLTransitionSequence:
        """
        :return: sequence of transitions. Note that the last action is a possible action but doesn't have to be taken
        """
        if self._t == 0 and self._collect_history:
            self.episode_states = [self._next_state]
            self.episode_actions = []
            self.episode_rewards = []
        states = []
        actions = []
        rewards = []
        terminal_states = []
        self._next_action = self._choose_action(self._next_state, self._t)
        step = 0
        while num_steps < 0 or step < num_steps:
            state = self._next_state
            action = self._next_action
            self._next_state, reward, is_terminal, _ = self._env.step(action)
            self.sample_count += 1

            self._episode_reward += self.discount_factor ** self._t * reward
            self._episode_score += reward
            self._t += 1

            self._next_action = self._choose_action(self._next_state, self._t)
            if not self._evaluation_mode:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                terminal_states.append(is_terminal)
                step += 1

            if is_terminal or self._t >= self._maxlen > 0:
                self.reward_ema = (1 - self._reward_log_smoothing) * self.reward_ema + \
                                  self._reward_log_smoothing * self._episode_reward
                self.score_ema = (1 - self._reward_log_smoothing) * self.score_ema + \
                                 self._reward_log_smoothing * self._episode_score

                states.append(self._next_state)
                actions.append(self._next_action)
                self._next_state = self._env.reset()
                self._next_action = self._choose_action(self._next_state, 0)
                self._episode_reward = 0.
                self._episode_score = 0.
                self._episode += 1
                self._t = 0

                break

        if len(states) == len(rewards):
            states.append(self._next_state)
            actions.append(self._next_action)

        if self._collect_history:
            self.episode_states.extend(states[1:])
            self.episode_actions.extend(actions[:-1])
            self.episode_rewards.extend(rewards)

        return data.RLTransitionSequence(
            rewards=np.array(rewards, np.float32),
            transition_sequence=data.TransitionSequence(
                states=[np.array(state, dtype=np.float32) for state in states],
                actions=np.array(actions, self._policy.action_type),
                is_terminal=np.array([is_terminal], dtype=np.float32)
            )
        )


def collect_batch(trainer: Trainer, sequence_length: int, batch_size: int) -> data.Batch[data.RLTransitionSequence]:
    return data.Batch([trainer.collect_sequence(sequence_length) for _ in range(batch_size)], trainer.discount_factor)


def rl_eval_range(start: int, end: int, trainers: Sequence[Trainer], tester: Optional[Trainer],
                  eval_frequency: Optional[int], return_score: bool=False):
    with tqdm.tqdm(range(start, end)) as pbar:
        for iteration in range(start, end):
            sample_count = sum(trainer.sample_count for trainer in trainers)
            if tester is not None and iteration % eval_frequency == eval_frequency - 1:
                tester.collect_sequence()
                visualization.reporting.global_summary_writer.add_scalar('evaluation reward', tester.reward_ema, sample_count)
                visualization.reporting.global_summary_writer.add_scalar('evaluation score', tester.score_ema, sample_count)
            reward_emas = [trainer.reward_ema for trainer in trainers]
            visualization.reporting.global_summary_writer.add_scalar('training reward', np.mean(reward_emas), sample_count)
            score_emas = [trainer.score_ema for trainer in trainers]
            visualization.reporting.global_summary_writer.add_scalar('training score', np.mean(score_emas), sample_count)
            pbar.update(1)
            test_score = '-' if tester is None else tester.score_ema
            pbar.set_description(f"r: {np.mean(score_emas)}/{test_score}, "
                                 f"iteration: {iteration}, samples: {sample_count}")
            if return_score:
                yield iteration, np.mean(score_emas)
            else:
                yield iteration

