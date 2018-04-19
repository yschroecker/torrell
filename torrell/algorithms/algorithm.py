import abc
from typing import Optional, Sequence

import environments.environment
import trainers.data_collection

from torrell import policies


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, trainers_: Sequence[trainers.data_collection.Trainer],
                 trainer_config: trainers.data_collection.TrainerConfig,
                 policy: policies.policy.Policy):
        self._trainers = trainers_
        self.policy = policy
        self._trainer_config = trainer_config

    @abc.abstractmethod
    def iterate(self, iteration: int):
        pass

    def rl_eval_range(self, start: int, end: int, test_env: Optional[environments.environment.Environment]=None,
                      eval_frequency: Optional[int]=None, return_score: bool=False):
        if test_env is None:
            tester = None
        else:
            tester = trainers.data_collection.Trainer(test_env, self.policy, self._trainer_config)
        return trainers.data_collection.rl_eval_range(start, end, self._trainers, tester, eval_frequency, return_score)

