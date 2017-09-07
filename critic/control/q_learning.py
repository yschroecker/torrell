import torch

import critic.temporal_difference
import torch_util


class DiscreteQLearning(critic.temporal_difference.TemporalDifference):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, target_update_rate: int):
        super().__init__(model, optimizer, target_update_rate)
        self._loss_fn = torch.nn.MSELoss()

    def update(self, batch: critic.temporal_difference.Batch):
        states = torch.autograd.Variable(batch.states, requires_grad=False)
        actions = torch.autograd.Variable(batch.actions, requires_grad=False)
        intermediate_returns = torch.autograd.Variable(batch.intermediate_returns, requires_grad=False)
        bootstrap_weights = torch.autograd.Variable(batch.bootstrap_weights, requires_grad=False)
        bootstrap_states = torch.autograd.Variable(batch.bootstrap_states, volatile=True)

        q_values = self._online_network(states)
        values = q_values.gather(dim=1, index=actions.unsqueeze(1))

        next_q_values = self._target_network(bootstrap_states)
        # noinspection PyArgumentList
        next_values = torch.max(next_q_values, dim=1)[0]
        target_values = intermediate_returns + bootstrap_weights*next_values
        target_values.volatile = False
        loss = self._loss_fn(values, target_values)

        self._update(loss)
