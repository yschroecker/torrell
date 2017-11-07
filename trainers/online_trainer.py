from typing import Any, Type, Tuple, NamedTuple, Sequence, Callable, Generic

import torch.optim
import torch.optim.lr_scheduler
import tqdm
import numpy as np

from actor.actor_base import Actor
from environments.environment import Environment, ActionT
from critic.temporal_difference import TemporalDifferenceBase, Batch
from optimization_strategies.optimization_strategy import OptimizationStrategy
from policies.policy import PolicyModel, Policy
import visualization


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


class DiscreteTrainerBase:
    def __init__(self, trainer_config: TrainerConfig):
        self._optimization_strategy = trainer_config.optimization_strategy
        self._reward_clipping = trainer_config.reward_clipping

        self.reward_ema = 0
        self.eval_score_ema = 0
        self.eval_reward_ema = 0
        self._sample_count = 0
        self._hooks = trainer_config.hooks

    def do_train(self, iteration: int, batch: Batch) -> str:
        for hook_iter, hook in self._hooks:
            if iteration % hook_iter == 0:
                hook(iteration)
        self._sample_count += batch.states.shape[0]
        self._optimization_strategy.iterate(iteration, batch)
        return f"r: {self.reward_ema}/{self.eval_reward_ema}, iteration: {iteration}, samples: {self._sample_count}"


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

    def _end_evaluation(self):
        self._evaluation_countdown = self._evaluation_frequency
        self._evaluation_mode = False
        self._env.treat_life_lost_as_terminal = True

    def _start_evaluation(self):
        self._evaluation_mode = True
        self._env.treat_life_lost_as_terminal = False  # TODO: refactor!!!

    def _choose_action(self, state: np.ndarray, t: int) -> ActionT:
        return self._policy.sample(state, t, not self._evaluation_mode)

    def collect_transitions(self, num_steps: int) -> \
            Tuple[Sequence[Any], Sequence[ActionT], Sequence[float], Sequence[float], Sequence[Any], Sequence[ActionT]]:
        states = []
        actions = []
        rewards = []
        terminal_states = []
        next_states = []
        next_actions = []
        self._next_action = self._choose_action(self._next_state, 0)
        step = 0
        while step < num_steps:
            self._evaluation_countdown -= 1
            state = self._next_state
            action = self._next_action
            self._next_state, reward, is_terminal, _ = self._env.step(action)
            if not self._evaluation_mode:
                # noinspection PyTypeChecker
                reward = np.clip(reward, self._reward_clipping[0], self._reward_clipping[1])

            self._episode_reward += self._discount_factor ** self._t * reward
            self._episode_score += reward
            self._t += 1

            if not self._evaluation_mode:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                terminal_states.append(is_terminal)
                next_states.append(self._next_state)
                next_actions.append(self._next_action)
                step += 1

            if is_terminal or self._t >= self._maxlen > 0:
                summary_target = 'evaluation reward' if self._evaluation_mode else 'episode reward'
                visualization.global_summary_writer.add_scalar(summary_target, self._episode_reward, self._episode)
                summary_target = 'evaluation score' if self._evaluation_mode else 'episode score'
                visualization.global_summary_writer.add_scalar(summary_target, self._episode_score, self._episode)
                if self._evaluation_mode:
                    self.eval_reward_ema = (1 - self._reward_log_smoothing) * self.eval_reward_ema + \
                        self._reward_log_smoothing * self._episode_reward
                    self.eval_score_ema = (1 - self._reward_log_smoothing) * self._episode_score + \
                        self._reward_log_smoothing * self._episode_score

                else:
                    self.reward_ema = (1 - self._reward_log_smoothing) * self.reward_ema + \
                                      self._reward_log_smoothing * self._episode_reward
                self._next_state = self._env.reset()
                self._episode_reward = 0.
                self._episode_score = 0.
                self._episode += 1
                if self._evaluation_mode:
                    self._end_evaluation()
                elif self._evaluation_countdown <= 0 < self._evaluation_frequency:
                    # noinspection PyAttributeOutsideInit
                    self._start_evaluation()
                self._t = 0
            self._next_action = self._choose_action(self._next_state, self._t)
        return states, actions, rewards, terminal_states, next_states, next_actions


class DiscreteOnlineTrainer(DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: TrainerConfig, batch_size: int = 1):
        self._batch_size = batch_size
        super().__init__(env, trainer_config)

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            states, actions, rewards, terminal_states, next_states, next_actions = \
                self.collect_transitions(self._batch_size)

            bootstrap_weights = self._discount_factor * (1 - np.array(terminal_states, dtype=np.float32))
            # noinspection PyUnresolvedReferences
            batch = Batch(
                states=np.array(states, dtype=np.float32),
                actions=np.array(actions, dtype=self._action_type),
                intermediate_returns=np.array(rewards, dtype=np.float32),
                bootstrap_weights=bootstrap_weights,
                bootstrap_states=np.array(next_states, dtype=np.float32),
                bootstrap_actions=np.array(next_actions, dtype=self._action_type)
            )
            trange.set_description(self.do_train(iteration, batch))


class DiscreteNstepTrainer(DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: TrainerConfig, batch_size: int = 1):
        self._batch_size = batch_size
        super().__init__(env, trainer_config)

    def get_batch(self) -> Batch:
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
                current_return = rewards[i] + self._discount_factor * current_return
            intermediate_returns.append(current_return)
            bootstrap_states.append(final_state)
            bootstrap_actions.append(final_action)

        bootstrap_weights = np.array(bootstrap_weights[::-1], dtype=np.float32)
        bootstrap_states = bootstrap_states[::-1]
        bootstrap_actions = bootstrap_actions[::-1]
        intermediate_returns = intermediate_returns[::-1]

        # noinspection PyUnresolvedReferences
        batch = Batch(
            states=[np.array(state, dtype=np.float32) for state in states],
            actions=np.array(actions, dtype=self._action_type),
            intermediate_returns=np.array(intermediate_returns, dtype=np.float32),
            bootstrap_weights=np.array(bootstrap_weights, dtype=np.float32),
            bootstrap_states=[np.array(bootstrap_state, dtype=np.float32) for bootstrap_state in bootstrap_states],
            bootstrap_actions=np.array(bootstrap_actions, dtype=self._action_type)
        )
        return batch

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            batch = self.get_batch()
            trange.set_description(self.do_train(iteration, batch))
