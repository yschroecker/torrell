from typing import Any, TypeVar, Union, Tuple

import numpy as np
import torch

FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]

TensorT = TypeVar('TensorT', FloatTensor, LongTensor)


def load_input(cuda: bool, input_array: np.ndarray) -> TensorT:
    if input_array.dtype == np.float32:
        tensor = torch.from_numpy(input_array).type(torch.FloatTensor)
    elif input_array.dtype == np.int32:
        tensor = torch.from_numpy(input_array).type(torch.LongTensor)
    else:
        assert False, 'unknown input type'

    if cuda:
        return tensor.cuda()
    else:
        return tensor


def load_inputs(cuda: bool, *input_arrays: np.ndarray) -> Tuple[Any, ...]:
    return tuple(load_input(cuda, input_array) for input_array in input_arrays)


def module_is_cuda(target_module: torch.nn.Module) -> bool:
    return next(target_module.parameters()).is_cuda


def rcumsum(tensor: TensorT) -> TensorT:
    reverse_indices = torch.arange(tensor.size(0) - 1, -1, -1).long()
    if tensor.is_cuda:
        reverse_indices = reverse_indices.cuda()
    return tensor[reverse_indices].cumsum(dim=0)[reverse_indices]
