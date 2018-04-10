import torch

import critic.temporal_difference
import visualization
import data


class ValueTD(critic.temporal_difference.ValueTemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, target_update_rate: int, *args, **kwargs):
        super().__init__(model, target_update_rate, *args, **kwargs)

    def _update(self, batch: data.Batch[data.TensorRLTransitionSequence]) -> torch.autograd.Variable:
        states = torch.autograd.Variable(batch.states(), requires_grad=False)
        intermediate_returns = torch.autograd.Variable(batch.intermediate_returns(), requires_grad=False)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights(), requires_grad=False)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states(), volatile=True)

        values_o = self._online_network(states).squeeze()
        next_values_t = self._target_network(bootstrap_states).squeeze()

        target_values = intermediate_returns + bootstrap_weights * next_values_t
        visualization.global_summary_writer.add_scalar(
            f'TD/target_values ({self.name})', target_values.mean().data[0], self._update_counter
        )
        target_values.volatile = False
        # import numpy as np
        # if np.any(intermediate_returns.data.cpu().numpy() > 0):
            # import pdb;pdb.set_trace()

        return torch.mean((values_o.squeeze() - target_values) ** 2)

    def values(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return self._online_network(states).squeeze()


class QValueTD(critic.temporal_difference.TemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, target_update_rate: int, *args, **kwargs):
        super().__init__(model, target_update_rate, *args, **kwargs)

    def _update(self, batch: critic.temporal_difference.TensorBatch) -> torch.autograd.Variable:
        states = torch.autograd.Variable(batch.states, requires_grad=False)
        actions = torch.autograd.Variable(batch.actions, requires_grad=False)
        intermediate_returns = torch.autograd.Variable(batch.intermediate_returns, requires_grad=False)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights, requires_grad=False)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states, volatile=True)
        bootstrap_actions = torch.autograd.Variable(batch.bootstrap_actions, volatile=True)

        q_values_o = self._online_network(states)
        values_o = q_values_o.gather(dim=1, index=actions.unsqueeze(1))
        next_q_values_t = self._target_network(bootstrap_states)
        next_values_t = next_q_values_t.gather(dim=1, index=bootstrap_actions.unsqueeze(1)).squeeze()

        target_values = intermediate_returns + bootstrap_weights * next_values_t
        visualization.global_summary_writer.add_scalar(
            f'TD/target_values ({self.name})', target_values.mean().data[0], self._update_counter
        )
        target_values.volatile = False
        return torch.mean((values_o.squeeze() - target_values) ** 2)

    def values(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        return self._online_network(states).squeeze()
