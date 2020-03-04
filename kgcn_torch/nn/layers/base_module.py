import torch.nn as nn

from .init import Initializer, feedforward_init


class BaseModule(nn.Module):
    """ BaseModule for pytorch model.
    """
    def _initialize(self, initializer: Initializer):
        feedforward_init(self, initializer)

    def forward(self):
        raise NotImplementedError("")
