import torch
import numpy as np


class SimpleConvQNetwork(torch.nn.Module):
    def __init__(self, input_width: int, input_height: int, input_history: int, num_actions: int):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(input_history, 32, 8, 4, 4)
        self._relu1 = torch.nn.ReLU()
        self._conv2 = torch.nn.Conv2d(32, 64, 4, 2, 2)
        self._relu2 = torch.nn.ReLU()
        self._conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self._relu3 = torch.nn.ReLU()
        width = ((input_width//4 + 1)//2 + 1)
        height = ((input_height//4 + 1)//2 + 1)
        self._linear1 = torch.nn.Linear(width * height * 64, 512)
        self._relu4 = torch.nn.ReLU()
        self._linear2 = torch.nn.Linear(512, num_actions)

    def forward(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        x = self._relu1(self._conv1(states))
        x = self._relu2(self._conv2(x))
        x = self._relu3(self._conv3(x))
        x = self._relu4(self._linear1(x.view(x.size(0), -1)))
        return self._linear2(x)
