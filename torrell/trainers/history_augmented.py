from typing import NamedTuple, Sequence

import numpy as np

import data
import trainers.data_collection


class History(NamedTuple):
    states: Sequence[np.ndarray]
    actions: Sequence[np.ndarray]


class RLTransitionSequenceWithHistory(NamedTuple):
    history: History
    transition_sequence: data.RLTransitionSequence

    @property
    def states(self) -> Sequence[data.State]:
        return self.transition_sequence.states

    @property
    def actions(self) -> np.ndarray:
        return self.transition_sequence.actions

    @property
    def rewards(self) -> np.ndarray:
        return self.transition_sequence.rewards

    @property
    def is_terminal(self) -> np.ndarray:
        return self.transition_sequence.is_terminal

    @property
    def size(self) -> int:
        return self.transition_sequence.size

    def to_tensor(self, use_cuda: bool) -> 'RLTransitionSequenceWithHistory':  # TODO: types are wrong
        return RLTransitionSequenceWithHistory(self.history, self.transition_sequence.to_tensor(use_cuda))


def collect_sequence_with_history(trainer: trainers.data_collection.Trainer, sequence_length: int) -> \
        RLTransitionSequenceWithHistory:
    sequence = trainer.collect_sequence(sequence_length)
    history = History(trainer.episode_states[:], trainer.episode_actions[:])  # TODO: profile
    return RLTransitionSequenceWithHistory(history, sequence)


def collect_batch_with_history(trainer: trainers.data_collection.Trainer, sequence_length: int, batch_size: int) -> \
        data.Batch[RLTransitionSequenceWithHistory]:
    return data.Batch([collect_sequence_with_history(trainer, sequence_length) for _ in range(batch_size)],
                      trainer.discount_factor)

