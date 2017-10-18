from typing import Optional

import numpy as np
import tqdm

import trainers.online_trainer
import trainers.ring_buffer
from critic.temporal_difference import Batch
from environments.environment import Environment


class DiscreteExperienceReplay(trainers.online_trainer.DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: trainers.online_trainer.TrainerConfig, memory_size: int,
                 batch_size: int, initial_population: int, transition_batch_size: int=1):
        super().__init__(env, trainer_config)
        self._batch_size = batch_size
        self._buffers = trainers.ring_buffer.RingBufferCollection(
            memory_size, [trainer_config.state_dim, 1, 1, 1, trainer_config.state_dim, 1],
            dtypes=[np.float32, np.int32, np.float32, np.float32, np.float32, np.int32]
        )
        self._initial_population = initial_population
        self._transition_batch_size = transition_batch_size

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            self._iterate(iteration, trange)

    def _iterate(self, iteration: int, trange: tqdm.tqdm):
        states, actions, rewards, terminal_states, next_states, next_actions = \
            self.collect_transitions(self._transition_batch_size)

        bootstrap_weights = self._discount_factor * (1 - np.array(terminal_states, dtype=np.float32))
        self._buffers.extend(states, actions, rewards, bootstrap_weights, next_states, next_actions)

        for _ in range(self._transition_batch_size):
            if self._buffers.size >= self._initial_population:
                states, actions, rewards, bootstrap_weights, next_states, next_actions = \
                    self._buffers.sample(self._batch_size)

                # noinspection PyUnresolvedReferences
                batch = Batch(
                    states=states,
                    actions=actions.squeeze(),
                    intermediate_returns=rewards.squeeze(),
                    bootstrap_weights=bootstrap_weights.squeeze(),
                    bootstrap_states=next_states,
                    bootstrap_actions=next_actions.squeeze(),
                    importance_weights=self._get_importance_weights(states, actions.squeeze())
                )
                trange.set_description(self.do_train(iteration, batch))

    def _get_importance_weights(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        return None
