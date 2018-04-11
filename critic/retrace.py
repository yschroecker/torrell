import numpy as np
import torch

import policies.policy
import critic.temporal_difference
import data
import torch_util
import visualization
from scripts.tabular.sagil import train_supervised


class Retrace(critic.temporal_difference.ValueTemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, sample_policy: policies.policy.PolicyModel,
                 current_policy: policies.policy.PolicyModel,
                 lambda_decay: float, *args, use_subsequences: bool = True,
                 state_distribution: torch.nn.Module = None, **kwargs):
        super().__init__(model, 1, *args, **kwargs)
        self._lambda_decay = lambda_decay
        self._sample_policy = sample_policy
        self._current_policy = current_policy
        self._use_subsequences = use_subsequences
        self._state_distribution = state_distribution

    def _update(self, batch: data.Batch[data.TensorRLTransitionSequence]) -> torch.autograd.Variable:
        td_loss = []
        zero = torch.autograd.Variable(torch.zeros((1,)), requires_grad=False)
        if self._sample_policy.is_cuda:
            zero = zero.cuda()
        for sequence in batch.sequences:
            states = torch.autograd.Variable(sequence.states, requires_grad=False)
            actions = torch.autograd.Variable(sequence.actions, requires_grad=False)
            reward = torch.autograd.Variable(sequence.rewards, requires_grad=False)
            is_terminal = torch.autograd.Variable(sequence.is_terminal, requires_grad=False)
            log_iw = (self._current_policy.log_probability(states[:-1], actions[:-1]) -
                      self._sample_policy.log_probability(states[:-1], actions[:-1])).squeeze()
            log_iw = torch.max(zero, log_iw)

            iws = torch.exp(torch_util.rcumsum(log_iw))

            ess = ((iws.sum()**2)/(iws**2).sum()).data
            if self._current_policy.is_cuda:
                ess = ess.cpu()
            visualization.reporting.global_summary_writer.add_scalar("Retrace ESS", ess.numpy())
            values = self._online_network(states).squeeze()
            # values[-1] = values[-1] * (1 - is_terminal)
            discount_weights = torch.pow(batch.discount_factor, torch.arange(0, sequence.rewards.size(0)))
            decay_weights = torch.pow(self._lambda_decay, torch.arange(0, sequence.rewards.size(0)))
            combined_weights = torch.autograd.Variable(discount_weights * decay_weights, requires_grad=False)
            if self._sample_policy.is_cuda:
                combined_weights = combined_weights.cuda()
            td_error = reward + batch.discount_factor * values[1:] - values[:-1] - is_terminal * batch.discount_factor * values[-1]
            cumulative_td_error = torch_util.rcumsum(td_error * iws * combined_weights)/combined_weights
            if self._state_distribution is not None:
                state_iws = self._state_distribution.state_weights(batch)
                td_loss.append(state_iws * ((cumulative_td_error + values[:-1]).detach() - values[:-1])**2)
            else:
                td_loss.append(((cumulative_td_error + values[:-1]).detach() - values[:-1])**2)
            '''
            for t in range(len(sequence.rewards) - 1, -1, -1):
                if t == len(sequence.rewards) - 1:
                    target_value = (1 - is_terminal) * batch.discount_factor * values[t + 1]
                else:
                    target_value = batch.discount_factor * values[t + 1]
                target_value = target_value + reward[t]
                online_value = values[t]
                sequence_loss = self._lambda_decay * iws[t] * ((sequence_loss + target_value).detach() - online_value)
                td_loss.append(sequence_loss**2/iws[t])
            '''


            '''
            current_sequence = sequence
            while current_sequence.rewards.size(0) >= 1:
                lambda_decay = torch.autograd.Variable(
                    torch_util.load_input(self._current_policy.is_cuda,
                                          self._lambda_decay ** np.arange(current_sequence.rewards.size(0)).astype(np.float32)),
                    requires_grad=False)
                sequence_states = torch.autograd.Variable(current_sequence.states, requires_grad=False)
                sequence_actions = torch.autograd.Variable(current_sequence.actions, requires_grad=False)

                discount_weights = torch.pow(batch.discount_factor, torch.arange(1,
                                                                                 current_sequence.states.size(0)))
                if self._sample_policy.is_cuda:
                    discount_weights = discount_weights.cuda()
                sequence_discount_weights = torch.autograd.Variable(discount_weights, requires_grad=False)
                sequence_rewards = torch.autograd.Variable(current_sequence.rewards, requires_grad=False)
                sequence_is_terminal = torch.autograd.Variable(current_sequence.is_terminal, requires_grad=False)
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
                if current_sequence.rewards.size(0) > 1:
                    current_sequence = current_sequence.tail()
                else:
                    break
        '''
        return torch.cat(td_loss).mean()

    def values(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return self._online_network(states).squeeze()
