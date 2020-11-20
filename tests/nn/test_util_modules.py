import torch
import torch.nn as nn
import pytest

from kgcn_torch.nn.util_modules import Concatenate


@pytest.mark.parametrize('axis, expected', [
    (0, [10, 4, 3, 2]),
    (1, [5, 8, 3, 2]),
    (2, [5, 4, 6, 2]),
    (3, [5, 4, 3, 4]),        
])
def test_concatenate(axis, expected):
    a = torch.randn(5, 4, 3, 2)
    b = torch.randn(5, 4, 3, 2)
    net = Concatenate()
    c = net(a, b, axis=axis)
    assert list(c.shape) == expected

    
