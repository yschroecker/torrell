from typing import Generic, TypeVar, List, NamedTuple, Union, Sequence

import numpy as np
import torch

import torch_util


State = Union[np.ndarray, List[np.ndarray]]
ActionTensor = Union[torch_util.FloatTensor, torch_util.LongTensor]


class TensorTransitionSequence(NamedTuple):
    states: torch.FloatTensor
    actions: ActionTensor
    is_terminal: torch.FloatTensor

    def tail(self) -> 'TensorTransitionSequence':
        return TensorTransitionSequence(
            states=self.states[1:],
            actions=self.actions[1:],
            is_terminal=self.is_terminal,
        )


class TransitionSequence(NamedTuple):
    states: Sequence[State]
    actions: np.ndarray
    is_terminal: np.ndarray

    def to_tensor(self, use_cuda: bool) -> TensorTransitionSequence:
        if type(self.states) is list:
            states = np.array(self.states)
        else:
            states = self.states

        return TensorTransitionSequence(
            *torch_util.load_inputs(use_cuda, states, self.actions, self.is_terminal)
        )

    TensorType = TensorTransitionSequence


class TensorRLTransitionSequence(NamedTuple):
    rewards: torch.FloatTensor
    transition_sequence: TensorTransitionSequence

    @property
    def states(self) -> torch.FloatTensor:
        return self.transition_sequence.states

    @property
    def actions(self) -> ActionTensor:
        return self.transition_sequence.actions

    @property
    def is_terminal(self) -> torch.FloatTensor:
        return self.transition_sequence.is_terminal

    def tail(self) -> 'TensorRLTransitionSequence':
        return TensorRLTransitionSequence(
            rewards=self.rewards[1:],
            transition_sequence=self.transition_sequence.tail()
        )


class RLTransitionSequence(NamedTuple):
    rewards: np.ndarray
    transition_sequence: TransitionSequence

    @property
    def states(self) -> Sequence[State]:
        return self.transition_sequence.states

    @property
    def actions(self) -> np.ndarray:
        return self.transition_sequence.actions

    @property
    def is_terminal(self) -> np.ndarray:
        return self.transition_sequence.is_terminal

    def to_tensor(self, use_cuda: bool) -> TensorRLTransitionSequence:
        return TensorRLTransitionSequence(
            torch_util.load_input(use_cuda, self.rewards),
            self.transition_sequence.to_tensor(use_cuda)
        )

    TensorType = TensorRLTransitionSequence


class TensorOffPolicyTransitionSequence(NamedTuple):
    importance_weights: torch.FloatTensor
    rl_transition_sequence: TensorRLTransitionSequence


class OffPolicyTransitionSequence(NamedTuple):
    importance_weights: np.ndarray
    rl_transition_sequence: RLTransitionSequence

    def to_tensor(self, use_cuda: bool) -> TensorOffPolicyTransitionSequence:
        return TensorOffPolicyTransitionSequence(
            torch_util.load_input(use_cuda, self.importance_weights),
            self.rl_transition_sequence.to_tensor(use_cuda)
        )

    TensorType = TensorOffPolicyTransitionSequence


SequenceT = TypeVar('SequenceT')


class Batch(Generic[SequenceT]):
    def __init__(self, sequences: Sequence[SequenceT], discount_factor: float):
        self.sequences = sequences
        self.discount_factor = discount_factor

    def to_tensor(self, use_cuda: bool):
        return Batch([sequence.to_tensor(use_cuda) for sequence in self.sequences], self.discount_factor)

    '''
    def _cuda_cat(self, seq: Sequence[Union[torch_util.FloatTensor, torch_util.LongTensor]]) -> \
            Union[torch_util.FloatTensor, torch_util.LongTensor]:
        cat_seq = torch.cat(seq)
        if seq[0].is_cuda:
            return cat_seq.cuda()
        else:
    '''
    def size(self) -> int:
        return sum([len(sequence.rewards) for sequence in self.sequences])

    def states(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.states[:-1] for sequence in self.sequences])

    def actions(self) -> ActionTensor:
        return torch.cat([sequence.actions[:-1] for sequence in self.sequences])

    def bootstrap_states(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.states[1:] for sequence in self.sequences])

    def bootstrap_actions(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.actions[1:] for sequence in self.sequences])

    def bootstrap_weights(self) -> torch_util.FloatTensor:
        return torch.cat([weight.cuda() if sequence.rewards.is_cuda else weight for sequence in self.sequences
                          for weight in ([torch.pow(self.discount_factor,
                                                    torch.arange(0, sequence.rewards.size(0) - 1)),
                                          self.discount_factor ** (sequence.rewards.size(0) - 1) *
                                          (1 - sequence.is_terminal)]
                                         if sequence.rewards.size(0) > 1
                                         else [self.discount_factor * (1 - sequence.is_terminal)])])

    def intermediate_returns(self) -> torch_util.FloatTensor:
        def discounts(sequence):
            result = torch.pow(self.discount_factor, torch.arange(0, sequence.rewards.size(0)))
            if sequence.rewards.is_cuda:
                return result.cuda()
            return result
        tensor = torch.cat([torch_util.rcumsum(discounts(sequence) * sequence.rewards)/discounts(sequence)
                            for sequence in self.sequences])

        if self.sequences[0].rewards.is_cuda:
            return tensor.cuda()
        else:
            return tensor

