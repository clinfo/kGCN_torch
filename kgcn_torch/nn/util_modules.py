import torch
import torch.nn as nn


class Concatenate(torch.nn.Module):
    def __init__(self, axis=0):
        super(Concatenate, self).__init__()
        self.axis = axis

    def forward(self, *inputs):
        return torch.cat(inputs, axis=self.axis)
    
