import torch
import tensorboard

Tensor = torch.FloatTensor
ByteTensor = torch.ByteTensor
LongTensor = torch.LongTensor

global_summary_writer = tensorboard.SummaryWriter()
