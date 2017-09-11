import numpy as np
import tqdm

from critic.temporal_difference import Batch
import trainers.ring_buffer
import trainers.online_trainer


class DiscreteExperienceReplay(trainers.online_trainer.DiscreteTrainer):
    def __init__(self, trainer_config: trainers.online_trainer.TrainerConfig, memory_size: int,
                 batch_size: int, initial_population: int):
        super().__init__(trainer_config)
        self._batch_size = batch_size
        self._buffers = trainers.ring_buffer.RingBufferCollection(
            memory_size, [trainer_config.state_dim, 1, 1, 1, trainer_config.state_dim, 1],
            dtypes=[np.float32, np.int32, np.float32, np.float32, np.float32, np.int32]
        )
        self._initial_population = initial_population

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            states, actions, rewards, terminal_states, next_states, next_actions = \
                self.collect_transitions(1)

            bootstrap_weights = self._discount_factor * (1 - np.array(terminal_states, dtype=np.float32))
            self._buffers.append(states[0], actions[0], rewards[0], bootstrap_weights[0], next_states[0],
                                 next_actions[0])

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
                    bootstrap_actions=next_actions.squeeze()
                )
                trange.set_description(self._train(iteration, batch))

