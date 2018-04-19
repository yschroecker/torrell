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

    @property
    def size(self) -> int:
        return len(self.rewards)

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


def sequence_returns(reward_sequences: Sequence[torch_util.FloatTensor], discount_factor: float) -> \
        torch_util.FloatTensor:
    def discounts(sequence_rewards: torch_util.FloatTensor):
        result = torch.pow(discount_factor, torch.arange(0, sequence_rewards.size(0)))
        if sequence_rewards.is_cuda:
            return result.cuda()
        return result
    tensor = torch.cat([torch_util.rcumsum(discounts(sequence_rewards) * sequence_rewards)/discounts(sequence_rewards)
                        for sequence_rewards in reward_sequences])

    if reward_sequences[0].is_cuda:
        return tensor.cuda()
    else:
        return tensor


class Batch(Generic[SequenceT]):
    def __init__(self, sequences: Sequence[SequenceT], discount_factor: float):
        self.sequences = sequences
        self.discount_factor = discount_factor

    def to_tensor(self, use_cuda: bool):
        return Batch([sequence.to_tensor(use_cuda) for sequence in self.sequences], self.discount_factor)

    def size(self) -> int:
        return sum([sequence.size for sequence in self.sequences])

    def states(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.states[:-1] for sequence in self.sequences])

    def actions(self) -> ActionTensor:
        return torch.cat([sequence.actions[:-1] for sequence in self.sequences])

    def rewards(self) -> ActionTensor:
        return torch.cat([sequence.rewards for sequence in self.sequences])

    def bootstrap_states(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.states[-1:].expand_as(sequence.states[1:]) for sequence in self.sequences])

    def bootstrap_actions(self) -> torch_util.FloatTensor:
        return torch.cat([sequence.actions[-1:].expand_as(sequence.actions[1:]) for sequence in self.sequences])

    def bootstrap_weights(self) -> torch_util.FloatTensor:
        ranges = [torch.arange(sequence.rewards.size(0), 0, -1) for sequence in self.sequences]
        if self.sequences[0].rewards.is_cuda:
            ranges = [r.cuda() for r in ranges]
        return torch.cat([torch.pow((1 - sequence.is_terminal) * self.discount_factor, arange)
                          for sequence, arange in zip(self.sequences, ranges)])

    def intermediate_returns(self) -> torch_util.FloatTensor:
        return sequence_returns([sequence.rewards for sequence in self.sequences], self.discount_factor)


'''
def new_to_old_tensor(batch: Batch[TensorRLTransitionSequence]) -> critic.temporal_difference.TensorBatch:
    intermediate_returns = batch.intermediate_returns()
    importance_weights = torch.ones(intermediate_returns.size())
    if intermediate_returns.is_cuda:
        importance_weights = importance_weights.cuda()
    return critic.temporal_difference.TensorBatch(
        states=batch.states(),
        actions=batch.actions(),
        intermediate_returns=intermediate_returns,
        bootstrap_states=batch.bootstrap_states(),
        bootstrap_actions=batch.bootstrap_actions(),
        bootstrap_weights=batch.bootstrap_weights(),
        importance_weights=importance_weights
    )


def new_to_old_np(batch: Batch[RLTransitionSequence]) -> critic.temporal_difference.Batch:
    all_states = []
    all_actions = []
    all_intermediate_returns = []
    all_bootstrap_weights = []
    all_bootstrap_states = []
    all_bootstrap_actions = []
    for sequence in batch.sequences:
        states = sequence.states[:-1]
        actions = sequence.actions[:-1]
        rewards = sequence.rewards
        terminal_states = [False] * len(rewards)
        terminal_states[-1] = sequence.is_terminal
        next_states = sequence.states[1:]
        next_actions = sequence.actions[1:]

        final_state = next_states[-1]
        final_action = next_actions[-1]
        bootstrap_states = [final_state]
        bootstrap_actions = [final_action]
        bootstrap_weights = [batch.discount_factor * (1 - terminal_states[-1])]
        current_return = rewards[-1]
        intermediate_returns = [current_return]
        for i in range(len(states) - 2, -1, -1):
            if terminal_states[i]:
                final_state = next_states[i]
                final_action = next_actions[i]
                bootstrap_weights.append(0.)
                current_return = rewards[i]
            else:
                bootstrap_weights.append(bootstrap_weights[-1] * batch.discount_factor)
                current_return = rewards[i] + batch.discount_factor * current_return
            intermediate_returns.append(current_return)
            bootstrap_states.append(final_state)
            bootstrap_actions.append(final_action)

        bootstrap_weights = np.array(bootstrap_weights[::-1], dtype=np.float32)
        bootstrap_states = bootstrap_states[::-1]
        bootstrap_actions = bootstrap_actions[::-1]
        intermediate_returns = intermediate_returns[::-1]

        all_states.extend(states)
        all_actions.extend(actions)
        all_intermediate_returns.extend(intermediate_returns)
        all_bootstrap_weights.extend(bootstrap_weights)
        all_bootstrap_states.extend(bootstrap_states)
        all_bootstrap_actions.extend(bootstrap_actions)

    # noinspection PyUnresolvedReferences
    return critic.temporal_difference.Batch(
        states=[np.array(state, dtype=np.float32) for state in all_states],
        actions=np.array(all_actions, dtype=np.int32),
        intermediate_returns=np.array(all_intermediate_returns, dtype=np.float32),
        bootstrap_weights=np.array(all_bootstrap_weights, dtype=np.float32),
        bootstrap_states=[np.array(bootstrap_state, dtype=np.float32) for bootstrap_state in all_bootstrap_states],
        bootstrap_actions=np.array(all_bootstrap_actions, dtype=np.int32) # TODO: action type
    )

'''

