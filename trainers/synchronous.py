from typing import Sequence

import numpy as np
import tqdm

from environments.environment import Environment
from trainers.online_trainer import DiscreteNstepTrainer, DiscreteTrainerBase, TrainerConfig
from critic.temporal_difference import Batch


class SynchronizedDiscreteNstepTrainer(DiscreteTrainerBase):
    def __init__(self, envs: Sequence[Environment[int]], trainer_config: TrainerConfig,
                 look_ahead: int=1, batch_size: int=1):
        super().__init__(trainer_config)
        self._trainers = [DiscreteNstepTrainer(env, trainer_config if i == 0 else
            trainer_config._replace(evaluation_frequency=0), look_ahead) for i, env in enumerate(envs)]
        self._batch_size = batch_size

    def train(self, num_iterations: int):
        trange = tqdm.tqdm(total=num_iterations)
        iteration = 0
        while iteration < num_iterations:

            trainer_samples = [trainer.get_batch() for trainer in self._trainers]
            samples = trainer_samples[0]
            for trainer_batch in trainer_samples[1:]:
                samples = Batch(
                    np.concatenate((samples.states, trainer_batch.states)),
                    np.concatenate((samples.actions, trainer_batch.actions)),
                    np.concatenate((samples.intermediate_returns, trainer_batch.intermediate_returns)),
                    np.concatenate((samples.bootstrap_states, trainer_batch.bootstrap_states)),
                    np.concatenate((samples.bootstrap_actions, trainer_batch.bootstrap_actions)),
                    np.concatenate((samples.bootstrap_weights, trainer_batch.bootstrap_weights)),
                    None
                )

            indices = np.arange(samples.states.shape[0])
            np.random.shuffle(indices)

            for batch_start in range(0, samples.states.shape[0], self._batch_size):
                batch_indices = indices[batch_start:batch_start+self._batch_size]
                batch = Batch(
                    states=samples.states[batch_indices],
                    actions=samples.actions[batch_indices],
                    intermediate_returns=samples.intermediate_returns[batch_indices],
                    bootstrap_weights=samples.bootstrap_weights[batch_indices],
                    bootstrap_states=samples.bootstrap_states[batch_indices],
                    bootstrap_actions=samples.bootstrap_actions[batch_indices],
                )
                iteration += 1
                trange.update(1)
                self._reward_ema = self._trainers[0]._reward_ema
                self._eval_reward_ema = self._trainers[0]._eval_reward_ema
                trange.set_description(self.do_train(iteration, batch))
