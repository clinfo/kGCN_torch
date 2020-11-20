import torch
import torch.nn as nn


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, *inputs, axis=0):
        return torch.cat(inputs, axis=axis)
    
