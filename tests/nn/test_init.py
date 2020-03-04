import torch
import pytest

from kgcn_torch.nn.layers.init import feedforward_init, _feedforward_init, Initializer


@pytest.fixture(
    params=[
        Initializer.UNIFORM,
        Initializer.NORMAL,
        Initializer.XAVIER,
        Initializer.KAIMING,
        Initializer.XAVIER_UNIFORM,
        Initializer.XAVIER_NORMAL,
        Initializer.KAIMING_UNIFORM,
        Initializer.KAIMING_NORMAL,
    ]
)
def correct_initializer_case(request):
    return request.param


def test_feedfoward_init_with_correct_case(correct_initializer_case):
    net = torch.nn.Linear(3, 10)
    non_initialized_weight = torch.sum(net.weight.data).data
    net = feedforward_init(net, correct_initializer_case)
    assert torch.sum(net.bias.data).data == 0.0
    initialized_weight = torch.sum(net.weight.data).data
    assert non_initialized_weight != initialized_weight


def test__feedfoward_init_with_correct_case():
    pass
