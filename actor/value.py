import torch

import actor.actor_base
import torch_util


class EpsilonGreedy(actor.actor_base.DiscreteActor):
    def __init__(self, num_actions: int, q_model: torch.nn.Module,
                 initial_epsilon: float, final_epsilon: float, decay_rate: float):
        self.num_actions = num_actions

        self._final_epsilon = final_epsilon
        self._decay_delta = (initial_epsilon-final_epsilon)*decay_rate
        self._epsilon = initial_epsilon
        self._q_model = q_model

    def update(self):
        self._epsilon = max(self._epsilon - self._decay_delta, self._final_epsilon)

    def probabilities(self, states: torch.autograd.Variable, training: bool=True) -> torch_util.Tensor:
        epsilon = self._epsilon if training else 0

        q_values = self._q_model(states)
        # noinspection PyArgumentList
        _, argmax = torch.max(q_values, dim=1)
        batch_size = states.size()[0]
        probabilities: torch.FloatTensor = torch.ones((batch_size, self.num_actions))*epsilon/self.num_actions
        probabilities = probabilities.type(torch_util.Tensor)
        probabilities[torch.arange(0, batch_size).type(torch.LongTensor), argmax.data] += (1-epsilon)
        return probabilities


def greedy(num_actions: int, q_model: torch.nn.Module) -> EpsilonGreedy:
    return EpsilonGreedy(num_actions, q_model, 0, 0, 0)
