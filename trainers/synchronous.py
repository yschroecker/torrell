from typing import Sequence, TypeVar, Callable
import random

import trainers.data_collection
import data


T = TypeVar('T')


def merge_batches(batches: Sequence[data.Batch[T]]) -> data.Batch[T]:
    return data.Batch([sequence for batch in batches for sequence in batch.sequences],
                      batches[0].discount_factor)


class ParallelAgents:
    def __init__(self, trainers_: Sequence[trainers.data_collection.Trainer], steps_per_agent: int=1,
                 batch_size: int=1, collect_batch: Callable[[trainers.data_collection.Trainer, int, int],
                                                            data.Batch[T]]=trainers.data_collection.collect_batch):
        self._trainers = trainers_
        self._steps_per_agent = steps_per_agent
        self._batch_size = batch_size
        self._collect_batch = collect_batch

    def collect_transitions(self) -> data.Batch[T]:
        num_sample_states = 0
        batches = []
        while num_sample_states < self._batch_size:
            trainer = random.choice(self._trainers)
            trainer_batch = self._collect_batch(
                trainer, min(self._steps_per_agent, self._batch_size - num_sample_states), 1
            )

            batches.append(trainer_batch)
            num_sample_states += batches[-1].size()

        return merge_batches(batches)

