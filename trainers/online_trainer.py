from typing import Any, Type, Tuple, NamedTuple, Sequence, Callable, Generic, Optional
import abc
import time

import torch.optim
import torch.optim.lr_scheduler
import tqdm
import numpy as np

from actor.actor_base import Actor
from environments.environment import Environment, ActionT
from critic.temporal_difference import TemporalDifferenceBase
from optimization_strategies.optimization_strategy import OptimizationStrategy
from policies.policy import PolicyModel, Policy
import visualization
import data


class TrainerConfig(NamedTuple):
    state_dim: int
    optimization_strategy: OptimizationStrategy
    policy_model: PolicyModel[int]
    policy_builder: Callable[[PolicyModel], Policy]
    discount_factor: float
    reward_log_smoothing: float
    action_dim: int = 1
    reward_clipping: Tuple[float, float] = (-np.float('inf'), np.float('inf'))
    max_len: int = -1
    evaluation_frequency: int = -1
    hooks: Sequence[Tuple[int, Callable[[int], None]]] = []


class DiscreteTrainerBase(metaclass=abc.ABCMeta):
    def __init__(self, trainer_config: TrainerConfig):
        self._optimization_strategy = trainer_config.optimization_strategy
        self._reward_clipping = trainer_config.reward_clipping

        self.reward_ema = 0
        self.eval_score_ema = 0
        self.eval_reward_ema = 0
        self.discount_factor = trainer_config.discount_factor
        self._last_print = 0
        self._sample_count = 0
        self._hooks = trainer_config.hooks

    def do_train(self, iteration: int, batch: data.Batch[data.RLTransitionSequence]) -> str:
        for hook_iter, hook in self._hooks:
            if iteration % hook_iter == 0:
                hook(iteration)
        self._sample_count += sum(sequence.rewards.shape[0] for sequence in batch.sequences)
        self._optimization_strategy.iterate(iteration, batch)
        return f"r: {self.reward_ema}/{self.eval_reward_ema}, iteration: {iteration}, samples: {self._sample_count}"

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            samples = self.collect_transitions(iteration)
            if samples is not None:
                trange.set_description(self.do_train(iteration, samples))
                if time.time() - self._last_print > 60:
                    print(f"iteration {iteration}, eval_score {self.eval_score_ema}")
                    self._last_print = time.time()

    @abc.abstractmethod
    def collect_transitions(self, iteration: int) -> Optional[data.Batch[data.RLTransitionSequence]]:
        pass


class DiscreteTrainer(DiscreteTrainerBase, Generic[ActionT]):
    def __init__(self, env: Environment[ActionT], trainer_config: TrainerConfig):
        super().__init__(trainer_config)
        self._env = env
        self._policy = trainer_config.policy_builder(trainer_config.policy_model)
        self._discount_factor = trainer_config.discount_factor
        self._evaluation_frequency = trainer_config.evaluation_frequency
        self._action_type = trainer_config.policy_model.action_type

        self._end_evaluation()

        self._reward_log_smoothing = trainer_config.reward_log_smoothing
        self._maxlen = trainer_config.max_len
        self._t = 0
        self._episode_reward = 0.
        self._episode_score = 0.
        self._next_state = env.reset()
        self._next_action = self._choose_action(self._next_state, 0)
        self._episode = 0

        self.last_eval_score = 0
        self.num_samples = 0

    def _end_evaluation(self):
        self._evaluation_countdown = self._evaluation_frequency
        self._evaluation_mode = False
        self._env.treat_life_lost_as_terminal = True

    def _start_evaluation(self):
        self._evaluation_mode = True
        self._env.treat_life_lost_as_terminal = False  # TODO: refactor!!!

    def _choose_action(self, state: np.ndarray, t: int) -> ActionT:
        return self._policy.sample(state, t, not self._evaluation_mode)

    def collect_sequence(self, num_steps: int) -> data.RLTransitionSequence:
        states = [self._next_state]
        actions = [self._next_action]
        rewards = []
        is_terminal = False
        step = 0
        while step < num_steps and not is_terminal:
            self._evaluation_countdown -= 1
            state = self._next_state
            action = self._next_action
            self._next_state, reward, is_terminal, _ = self._env.step(action)
            if not self._evaluation_mode:
                self.num_samples += 1
                # noinspection PyTypeChecker
                reward = np.clip(reward, self._reward_clipping[0], self._reward_clipping[1])

            self._episode_reward += self._discount_factor ** self._t * reward
            self._episode_score += reward
            self._t += 1

            if not self._evaluation_mode:
                rewards.append(reward)
                step += 1

            if not self._evaluation_mode:
                states.append(self._next_state)
                actions.append(self._next_action)

            if is_terminal or self._t >= self._maxlen > 0:
                is_terminal = True
                summary_target = 'evaluation reward' if self._evaluation_mode else 'episode reward'
                visualization.global_summary_writer.add_scalar(summary_target, self._episode_reward, self._episode)
                summary_target = 'evaluation score' if self._evaluation_mode else 'episode score'
                visualization.global_summary_writer.add_scalar(summary_target, self._episode_score, self._episode)
                if self._evaluation_mode:
                    self.eval_reward_ema = (1 - self._reward_log_smoothing) * self.eval_reward_ema + \
                        self._reward_log_smoothing * self._episode_reward
                    self.eval_score_ema = (1 - self._reward_log_smoothing) * self._episode_score + \
                        self._reward_log_smoothing * self._episode_score
                    self.last_eval_score = self._episode_score

                else:
                    self.reward_ema = (1 - self._reward_log_smoothing) * self.reward_ema + \
                                      self._reward_log_smoothing * self._episode_reward
                self._episode_reward = 0.
                self._episode_score = 0.
                self._episode += 1
                self._next_state = self._env.reset()
                if self._evaluation_mode:
                    self._end_evaluation()
                    states = [self._next_state]
                    actions = [self._next_action]
                    rewards = []
                    is_terminal = False
                    step = 0
                elif self._evaluation_countdown <= 0 < self._evaluation_frequency:
                    # noinspection PyAttributeOutsideInit
                    self._start_evaluation()
                self._t = 0
            self._next_action = self._choose_action(self._next_state, self._t)

        assert len(rewards) > 0
        assert len(states) == len(rewards) + 1
        return data.RLTransitionSequence(
            rewards=np.array(rewards, dtype=np.float32),
            transition_sequence=data.TransitionSequence(
                states=[np.array(state, dtype=np.float32) for state in states],
                actions=np.array(actions, dtype=self._action_type),
                is_terminal=np.array([float(is_terminal)], dtype=np.float32)
            )
        )

    def _collect_transitions(self, batch_size: int, sequence_length: int = 1) -> data.Batch[data.RLTransitionSequence]:
        return data.Batch([self.collect_sequence(sequence_length) for _ in range(batch_size)])


class DiscreteOnlineTrainer(DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: TrainerConfig, batch_size: int = 1):
        self._batch_size = batch_size
        super().__init__(env, trainer_config)

    def collect_transitions(self, _) -> data.Batch[data.RLTransitionSequence]:
        return self._collect_transitions(self._batch_size, 1)


class DiscreteNstepTrainer(DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: TrainerConfig, batch_size: int = 1):
        self._batch_size = batch_size
        super().__init__(env, trainer_config)

    def collect_transitions(self, _) -> data.Batch[data.RLTransitionSequence]:
        return self._collect_transitions(1, self._batch_size)
