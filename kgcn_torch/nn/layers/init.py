from enum import Flag, auto
import torch.nn as nn


class Initializer(Flag):
    """ Initializer enum class.
    """

    UNIFORM = auto()
    NORMAL = auto()
    XAVIER = auto()  # XAVIER_NORMAL
    KAIMING = auto()  # KAIMING_NORMAL
    XAVIER_UNIFORM = XAVIER | UNIFORM
    XAVIER_NORMAL = XAVIER | NORMAL  # Glorot initialization
    KAIMING_UNIFORM = KAIMING | UNIFORM
    KAIMING_NORMAL = KAIMING | NORMAL


def feedforward_init(dnn: nn.Module, _init: Initializer) -> nn.Module:
    """ pytorch model initialization method.
    """
    if _init == Initializer.UNIFORM:
        _feedforward_init(dnn, False, False, False)
    elif _init == Initializer.NORMAL:
        _feedforward_init(dnn, False, False, True)
    elif _init == Initializer.XAVIER_UNIFORM:
        _feedforward_init(dnn, True, False, False)
    elif _init in [Initializer.XAVIER_NORMAL, Initializer.XAVIER]:
        _feedforward_init(dnn, True, False, True)
    elif _init == Initializer.KAIMING_UNIFORM:
        _feedforward_init(dnn, False, True, False)
    elif _init in [Initializer.KAIMING_NORMAL, Initializer.KAIMING]:
        _feedforward_init(dnn, False, True, True)
    elif _init is None:
        pass
    else:
        raise ValueError(f"Not supported input value = {_init}")
    return dnn


def _feedforward_init(
        dnn: nn.Module, init_xavier: bool = True, init_kaiming: bool = True, init_normal: bool = True):
    for name, param in dnn.named_parameters():
        if "bias" in name:
            param.data.zero_()
        if "weight" in name:
            if len(param.data.shape) == 1:
                if init_normal:
                    nn.init.normal_(param.data)
                else:
                    nn.init.uniform_(param.data)
                continue
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif init_kaiming:
                # not full supporting
                if init_normal:
                    nn.init.kaiming_normal_(param.data)
                else:
                    nn.init.kaiming_uniform_(param.data)
            else:
                if init_normal:
                    nn.init.normal_(param.data)
                else:
                    nn.init.uniform_(param.data)
