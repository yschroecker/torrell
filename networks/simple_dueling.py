import torch


class SimpleDuelingQNetwork(torch.nn.Module):
    def __init__(self, input_width: int, input_height: int, input_history: int, num_actions: int):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(input_history, 64, 8, 4, 4)
        self._relu1 = torch.nn.ReLU()
        self._conv2 = torch.nn.Conv2d(64, 128, 4, 2, 2)
        self._relu2 = torch.nn.ReLU()
        self._conv3 = torch.nn.Conv2d(128, 64, 3, 1, 1)
        self._relu3 = torch.nn.ReLU()
        width = ((input_width//4 + 1)//2 + 1)
        height = ((input_height//4 + 1)//2 + 1)
        self._linear_v_1 = torch.nn.Linear(width * height * 64, 512)
        self._relu_v = torch.nn.ReLU()
        self._linear_v_2 = torch.nn.Linear(512, 1)

        self._linear_a_1 = torch.nn.Linear(width * height * 64, 512)
        self._relu_a = torch.nn.ReLU()
        self._linear_a_2 = torch.nn.Linear(512, num_actions)

    def forward(self, states: torch.autograd.Variable) -> torch.autograd.Variable:
        x = self._relu1(self._conv1(states))
        x = self._relu2(self._conv2(x))
        x = self._relu3(self._conv3(x))

        v = self._relu_v(self._linear_v_1(x.view(x.size(0), -1)))
        v = self._linear_v_2(v)

        a = self._relu_a(self._linear_a_1(x.view(x.size(0), -1)))
        a = self._linear_a_2(a)

        return a - a.mean(dim=1, keepdim=True) + v.unsqueeze(1)
