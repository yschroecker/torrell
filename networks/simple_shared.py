import abc

import torch

import torch_util


class SimpleSharedNetwork(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, input_width: int, input_height: int, input_history: int, num_actions: int):
        super().__init__()
        '''
        self._conv1 = torch.nn.Conv2d(input_history, 16, 8, 4, 0)
        self._relu1 = torch.nn.ReLU()
        self._conv2 = torch.nn.Conv2d(16, 32, 4, 2, 0)
        self._relu2 = torch.nn.ReLU()
        width = ((input_width//4 + 1)//2 + 1)
        height = ((input_height//4 + 1)//2 + 1)
        num_linear = 256
        self._linear1 = torch.nn.Linear(2592, num_linear)
        self._relu3 = torch.nn.ReLU()
        '''
        num_linear = 512
        self._conv1 = torch.nn.Conv2d(input_history, 32, 8, 4, 4)
        self._relu1 = torch.nn.ReLU()
        self._conv2 = torch.nn.Conv2d(32, 64, 4, 2, 2)
        self._relu2 = torch.nn.ReLU()
        self._conv3 = torch.nn.Conv2d(64, 32, 3, 1, 1)
        self._relu3 = torch.nn.ReLU()
        # width = ((input_width // 4 + 1) // 2 + 1)
        # height = ((input_height // 4 + 1) // 2 + 1)
        self._linear1 = torch.nn.Linear(4608, num_linear)
        self._relu4 = torch.nn.ReLU()

        self._pi_out = torch.nn.Linear(num_linear, num_actions)
        torch.nn.init.normal(self._pi_out.weight, std=0.001)
        torch.nn.init.normal(self._pi_out.bias, std=0.001)
        self._v_out = torch.nn.Linear(num_linear, 1)

    def shared(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        x = self._relu1(self._conv1(states))
        x = self._relu2(self._conv2(x))
        x = self._relu3(self._conv3(x))
        x = self._relu4(self._linear1(x.view(x.size(0), -1)))
        # x = self._relu3(self._linear1(x.view(x.size(0), -1)))
        return x

    def v(self, states: torch_util.FloatTensor):
        return self._v_out(self.shared(states))

    def pi(self, states: torch_util.FloatTensor):
        return self._pi_out(self.shared(states))


class VNetwork(torch.nn.Module):
    def __init__(self, shared: torch.nn.Module):
        super().__init__()
        self._shared = shared

    def forward(self, states: torch_util.FloatTensor):
        return self._shared.v(states)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, shared: torch.nn.Module):
        super().__init__()
        self._shared = shared

    def forward(self, states: torch_util.FloatTensor):
        return self._shared.pi(states)
