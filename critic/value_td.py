from typing import Optional

import torch

import critic.temporal_difference
import visualization


class ValueTD(critic.temporal_difference.TemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, target_update_rate: int,
                 gradient_clip: Optional[float]=None, grad_report_rate: int=1000):
        super().__init__(model, optimizer, target_update_rate, gradient_clip, grad_report_rate)
        self._loss_fn = torch.nn.MSELoss()

    def _update(self, batch: critic.temporal_difference.TensorBatch) -> torch.autograd.Variable:
        states = torch.autograd.Variable(batch.states, requires_grad=False)
        intermediate_returns = torch.autograd.Variable(batch.intermediate_returns, requires_grad=False)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights, requires_grad=False)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states, volatile=True)

        values = self._online_network(states).squeeze()
        next_values = self._target_network(bootstrap_states).squeeze()

        target_values = intermediate_returns + bootstrap_weights*next_values
        visualization.global_summary_writer.add_scalar(
            f'TD/target_values ({self.name})', target_values.mean().data[0], self._update_counter
        )
        target_values.volatile = False
        loss = self._loss_fn(values, target_values)

        return loss

    def values(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return self._online_network(states).squeeze()
