from typing import Generic, TypeVar, Sequence, Optional, Type, Tuple

import numpy as np
import tqdm

import trainers.online_trainer
import trainers.ring_buffer
from critic.temporal_difference import Batch
from environments.environment import Environment


ActionT = TypeVar('ActionT')
SampleType = Tuple[np.ndarray, ActionT, float, float, np.ndarray, ActionT]


class FIFOReplayMemory(Generic[ActionT]):
    def __init__(self, state_dim: int, action_dim: int, action_type: Type[np.dtype], memory_size: int):
        self._buffers = trainers.ring_buffer.RingBufferCollection(
            memory_size, [state_dim, action_dim, 1, 1, state_dim, action_dim],
            dtypes=[np.float32, action_type, np.float32, np.float32, np.float32, action_type]
        )

    def sample(self, batch_size: int) -> Sequence[SampleType]:
        return self._buffers.sample(batch_size)

    def extend(self, states: Sequence[np.ndarray], actions: Sequence[ActionT], rewards: Sequence[float],
               bootstrap_weights: Sequence[float], booststrap_states: Sequence[np.ndarray],
               bootstrap_actions: Sequence[ActionT]):
        self._buffers.extend(states, actions, rewards, bootstrap_weights, booststrap_states, bootstrap_actions)

    @property
    def size(self):
        return self._buffers.size


class DiscreteExperienceReplay(trainers.online_trainer.DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: trainers.online_trainer.TrainerConfig, memory_size: int,
                 batch_size: int, initial_population: int, transition_batch_size: int=1):
        super().__init__(env, trainer_config)
        self._batch_size = batch_size
        self._memory = FIFOReplayMemory(trainer_config.state_dim, trainer_config.action_dim,
                                        trainer_config.policy_model.action_type, memory_size)
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
        self._memory.extend(states, actions, rewards, bootstrap_weights, next_states, next_actions)

        for _ in range(self._transition_batch_size):
            if self._memory.size >= self._initial_population:
                states, actions, rewards, bootstrap_weights, next_states, next_actions = \
                    self._memory.sample(self._batch_size)

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
