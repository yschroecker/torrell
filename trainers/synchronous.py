import time
from typing import Sequence

import numpy as np
import random
import tqdm

from critic.temporal_difference import Batch
from environments.environment import Environment
from trainers.online_trainer import DiscreteNstepTrainer, DiscreteTrainerBase, TrainerConfig


class SynchronizedDiscreteNstepTrainer(DiscreteTrainerBase):
    def __init__(self, envs: Sequence[Environment[int]], trainer_config: TrainerConfig,
                 look_ahead: int=1, batch_size: int=1):
        super().__init__(trainer_config)
        # noinspection PyProtectedMember
        self._trainers = [DiscreteNstepTrainer(
            env, trainer_config if i == 0
            else trainer_config._replace(evaluation_frequency=0), look_ahead) for i, env in enumerate(envs)]
        self._batch_size = batch_size
        self._last_print = 0
        assert batch_size % look_ahead == 0

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            samples = None
            while samples is None or samples.states.shape[0] < self._batch_size:
                trainer = random.choice(self._trainers)
                trainer_batch = trainer.get_batch()
                if samples is None:
                    samples = trainer_batch
                else:
                    samples = Batch(
                        np.concatenate((samples.states, trainer_batch.states)),
                        np.concatenate((samples.actions, trainer_batch.actions)),
                        np.concatenate((samples.intermediate_returns, trainer_batch.intermediate_returns)),
                        np.concatenate((samples.bootstrap_states, trainer_batch.bootstrap_states)),
                        np.concatenate((samples.bootstrap_actions, trainer_batch.bootstrap_actions)),
                        np.concatenate((samples.bootstrap_weights, trainer_batch.bootstrap_weights)),
                        None
                    )
            self.reward_ema = self._trainers[0].reward_ema
            self.eval_reward_ema = self._trainers[0].eval_reward_ema
            trange.set_description(self.do_train(iteration, samples))
            if time.time() - self._last_print > 60:
                print(f"iteration {iteration}, eval_score {self._trainers[0].eval_score_ema}")
                self._last_print = time.time()
