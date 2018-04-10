from typing import Any, TypeVar, Union, Tuple

import functools

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


@functools.lru_cache()
def _reverse_indices(tensor_size, cuda):
    reverse_indices = torch.arange(tensor_size - 1, -1, -1).long()
    if cuda:
        reverse_indices = reverse_indices.cuda()
    return reverse_indices


def rcumsum(tensor: TensorT) -> TensorT:
    reverse_indices = _reverse_indices(tensor.size(0), tensor.is_cuda)
    return tensor[reverse_indices].cumsum(dim=0)[reverse_indices]


def exclude_index(tensor: TensorT, index: int) -> TensorT:
    if index == 0:
        return tensor[1:]
    elif index == tensor.size(0) - 1:
        return tensor[:-1]
    else:
        return torch.cat([tensor[:index], tensor[index+1:]], dim=0)

