import numpy as np
import torch

import policies.policy
import critic.temporal_difference
import data
import torch_util
from scripts.tabular.sagil import train_supervised


class Retrace(critic.temporal_difference.ValueTemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, sample_policy: policies.policy.PolicyModel,
                 current_policy: policies.policy.PolicyModel,
                 lambda_decay: float, target_update_rate: int, *args, use_subsequences: bool = True, **kwargs):
        super().__init__(model, target_update_rate, *args, **kwargs)
        self._lambda_decay = lambda_decay
        self._sample_policy = sample_policy
        self._current_policy = current_policy
        self._use_subsequences = use_subsequences

    def _update(self, batch: data.Batch[data.TensorRLTransitionSequence]) -> torch.autograd.Variable:
        td_loss = []
        zero = torch.autograd.Variable(torch.zeros((1,)), requires_grad=False)
        if self._sample_policy.is_cuda:
            zero = zero.cuda()
        for sequence in batch.sequences:
            current_sequence = sequence
            while current_sequence.rewards.size(0) >= 1:
                lambda_decay = torch.autograd.Variable(
                    torch_util.load_input(self._current_policy.is_cuda,
                                          self._lambda_decay ** np.arange(sequence.rewards.size(0)).astype(np.float32)),
                    requires_grad=False)
                sequence_states = torch.autograd.Variable(sequence.states, requires_grad=False)
                sequence_actions = torch.autograd.Variable(sequence.actions, requires_grad=False)
                sequence_discount_weights = torch.autograd.Variable(sequence.discount_weights, requires_grad=False)
                sequence_rewards = torch.autograd.Variable(sequence.rewards, requires_grad=False)
                sequence_is_terminal = torch.autograd.Variable(sequence.is_terminal, requires_grad=False)
                log_iw = (self._current_policy.log_probability(sequence_states[:-1], sequence_actions[:-1]) -
                          self._sample_policy.log_probability(sequence_states[:-1], sequence_actions[:-1])).squeeze()
                if log_iw.size(0) > 1:
                    log_iw[1:] = torch.max(zero, log_iw[1:])
                retrace_iws = lambda_decay * torch.exp(log_iw).cumsum(dim=0)
                sequence_values = torch.cat([self._online_network(sequence_states[0:1])] +
                                            ([] if sequence_states.size(0) == 2 else
                                             [self._target_network(sequence_states[1:-1]).detach()]) +
                                            [self._target_network(sequence_states[-1:]).detach() *
                                             (1 - sequence_is_terminal)]).squeeze()
                discounted_values = torch.cat([sequence_values[0:1], sequence_discount_weights * sequence_values[1:]])
                if sequence_rewards.size(0) > 1:
                    discounted_rewards = torch.cat([sequence_rewards[0:1],
                                                    sequence_discount_weights[:-1] * sequence_rewards[1:]])
                else:
                    discounted_rewards = sequence_rewards
                td_error = retrace_iws*(discounted_rewards + discounted_values[1:] - discounted_values[:-1])

                td_loss.append(td_error.sum()**2/retrace_iws[0])
                current_sequence = current_sequence.tail()
        return torch.cat(td_loss).mean()

    def values(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return self._online_network(states).squeeze()
