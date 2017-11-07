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
    discount_weights: torch.FloatTensor


class TransitionSequence(NamedTuple):
    states: Sequence[State]
    actions: np.ndarray
    is_terminal: np.ndarray
    discount_weights: np.ndarray

    def to_tensor(self, use_cuda: bool) -> TensorTransitionSequence:
        if type(self.states) is list:
            states = np.array(self.states)
        else:
            states = self.states

        return TensorTransitionSequence(
            *torch_util.load_inputs(use_cuda, states, self.actions, self.is_terminal, self.discount_weights)
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

    @property
    def discount_weights(self) -> torch.FloatTensor:
        return self.transition_sequence.discount_weights


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

    @property
    def discount_weights(self) -> np.ndarray:
        return self.transition_sequence.discount_weights

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
    def __init__(self, sequences: Sequence[SequenceT]):
        self.sequences = sequences

    def to_tensor(self, use_cuda: bool):
        return Batch([sequence.to_tensor(use_cuda) for sequence in self.sequences])

    '''
    def _cuda_cat(self, seq: Sequence[Union[torch_util.FloatTensor, torch_util.LongTensor]]) -> \
            Union[torch_util.FloatTensor, torch_util.LongTensor]:
        cat_seq = torch.cat(seq)
        if seq[0].is_cuda:
            return cat_seq.cuda()
        else:
    '''

    def states(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.states[0:1] for sequence in self.sequences])

    def actions(self) -> ActionTensor:
        return torch.cat([sequence.actions[0:1] for sequence in self.sequences])

    def bootstrap_states(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.states[-1:] for sequence in self.sequences])

    def bootstrap_actions(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.actions[-1:] for sequence in self.sequences])

    def bootstrap_weights(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.discount_weights[-1:] * (1 - sequence.is_terminal) for sequence in self.sequences])

    def intermediate_returns(self) -> torch_util.FloatTensor:
        tensor = torch.FloatTensor([(sequence.discount_weights * sequence.rewards / sequence.discount_weights[0]).sum()
                                    for sequence in self.sequences])
        if self.sequences[0].discount_weights.is_cuda:
            return tensor.cuda()
        else:
            return tensor

