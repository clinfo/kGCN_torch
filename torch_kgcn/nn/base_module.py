from .utils import Initializer
from .init import feedforward_init


class BaseModule(nn.Module):
    def _initialize(self, initializer: Initializer):
        feedforward_init(self, initializer)
