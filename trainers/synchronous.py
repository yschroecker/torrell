import time
from typing import Sequence
import random

import numpy as np
import tqdm

from environments.environment import Environment
from trainers.online_trainer import DiscreteNstepTrainer, DiscreteTrainerBase, TrainerConfig
import data


class SynchronizedDiscreteNstepTrainer(DiscreteTrainerBase):
    def __init__(self, envs: Sequence[Environment[int]], trainer_config: TrainerConfig,
                 look_ahead: int=1, batch_size: int=1):
        super().__init__(trainer_config)
        # noinspection PyProtectedMember
        self._trainers = [DiscreteNstepTrainer(
            env, trainer_config if i == 0
            else trainer_config._replace(evaluation_frequency=0), look_ahead) for i, env in enumerate(envs)]
        self._batch_size = batch_size
        self._look_ahead = look_ahead
        assert batch_size % look_ahead == 0

    def collect_transitions(self, _) -> data.Batch[data.RLTransitionSequence]:
        samples = None
        while samples is None or sum(len(sequence.rewards) for sequence in samples.sequences) < self._batch_size:
            trainer = random.choice(self._trainers)
            trainer_batch = trainer.collect_sequence(self._look_ahead)
            if samples is None:
                samples = data.Batch([trainer_batch])
            else:
                samples = data.Batch(samples.sequences + [trainer_batch])
        self.reward_ema = self._trainers[0].reward_ema
        self.eval_reward_ema = self._trainers[0].eval_reward_ema
        return samples
