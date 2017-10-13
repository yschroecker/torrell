from typing import Optional
import abc

import torch

import critic.temporal_difference
import visualization


class DiscreteQLearningBase(critic.temporal_difference.TemporalDifferenceBase, metaclass=abc.ABCMeta):
    def __init__(self, model: torch.nn.Module, target_update_rate: int,
                 gradient_clip: Optional[float]=None, grad_report_rate: int=1000):
        super().__init__(model, target_update_rate, gradient_clip, grad_report_rate)
        self._loss_fn = torch.nn.MSELoss()

    @abc.abstractmethod
    def _target_values(self, states: torch.autograd.Variable, actions: torch.autograd.Variable,
                       intermediate_returns: torch.autograd.Variable, bootstrap_weights: torch.autograd.Variable,
                       bootstrap_states: torch.autograd.Variable) -> torch.autograd.Variable:
        pass

    def _update(self, batch: critic.temporal_difference.TensorBatch) -> torch.autograd.Variable:
        states = torch.autograd.Variable(batch.states, requires_grad=False)
        actions = torch.autograd.Variable(batch.actions, requires_grad=False)
        intermediate_returns = torch.autograd.Variable(batch.intermediate_returns, requires_grad=False)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights, requires_grad=False)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states, volatile=True)

        q_values_o = self._online_network(states)
        values_o = q_values_o.gather(dim=1, index=actions.unsqueeze(1))

        target_values = self._target_values(states, actions, intermediate_returns, bootstrap_weights, bootstrap_states)
        visualization.global_summary_writer.add_scalar(
            f'TD/online_values ({self.name})', values_o.mean().data[0], self._update_counter
        )
        visualization.global_summary_writer.add_scalar(
            f'TD/target_values ({self.name})', target_values.mean().data[0], self._update_counter
        )
        target_values.detach_()
        target_values.volatile = False
        loss = self._loss_fn(values_o, target_values)

        return loss


class DiscreteQLearning(DiscreteQLearningBase):
    def _target_values(self, states: torch.autograd.Variable, actions: torch.autograd.Variable,
                       intermediate_returns: torch.autograd.Variable, bootstrap_weights: torch.autograd.Variable,
                       bootstrap_states: torch.autograd.Variable) -> torch.autograd.Variable:
        next_q_values_t = self._target_network(bootstrap_states)
        # noinspection PyArgumentList
        next_values = torch.max(next_q_values_t, dim=1)[0]
        return intermediate_returns + bootstrap_weights*next_values


class DiscreteDoubleQLearning(DiscreteQLearningBase):
    def _target_values(self, states: torch.autograd.Variable, actions: torch.autograd.Variable,
                       intermediate_returns: torch.autograd.Variable, bootstrap_weights: torch.autograd.Variable,
                       bootstrap_states: torch.autograd.Variable) -> torch.autograd.Variable:
        next_q_values_t = self._target_network(bootstrap_states)
        next_q_values_o = self._online_network(bootstrap_states)
        # noinspection PyArgumentList
        argmax_actions = torch.torch.max(next_q_values_o, dim=1)[1]
        next_values = next_q_values_t.gather(dim=1, index=argmax_actions.unsqueeze(1)).squeeze()
        return intermediate_returns + bootstrap_weights*next_values
