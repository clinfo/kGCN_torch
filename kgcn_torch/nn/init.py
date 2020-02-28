from enum import Flag, auto
import torch.nn as nn


class Initializer(Flag):
    """ Initializer enum class.
    """
    UNIFORM = auto()
    NORMAL = auto()
    XAVIER = auto()             # XAVIER_NORMAL
    KAIMING = auto()            # KAIMING_NORMAL
    XAVIER_UNIFORM = XAVIER | UNIFORM
    XAVIER_NORMAL = XAVIER | NORMAL      # Glorot initialization
    KAIMING_UNIFORM = KAIMING | UNIFORM
    KAIMING_NORMAL = KAIMING | NORMAL


def feedforward_init(dnn: nn.Module, _init: Initializer) -> nn.Module:
    if _init == Initializer.UNIFORM:
        _feedforward_init(dnn, False, False, False)
        return dnn
    elif _init == Initializer.NORMAL:
        _feedforward_init(dnn, False, False, True)
        return dnn
    elif _init == Initializer.XAVIER_UNIFORM:
        _feedforward_init(dnn, True, False, False)
        return dnn
    elif _init in [Initializer.XAVIER_NORMAL, Initializer.XAVIER]:
        _feedforward_init(dnn, True, False, True)
        return dnn
    elif _init == Initializer.KAIMING_UNIFORM:
        _feedforward_init(dnn, False, True, False)
        return dnn
    elif _init in [Initializer.KAIMING_NORMAL, Initializer.KAIMING]:
        _feedforward_init(dnn, False, True, True)
        return dnn
    elif _init == None:
        pass
    else:
        raise ValueError(f"Not supported input value = {_init}")


def _feedforward_init(dnn: nn.Module,
                      init_xavier: bool=True,
                      init_kaiming: bool=True,
                      init_normal: bool=True):
    for name, p in dnn.named_parameters():
        if 'bias' in name:
            p.data.zero_()
        if 'weight' in name:
            if len(p.data.shape) == 1:
                if init_normal:
                    nn.init.normal_(p.data)
                else:
                    nn.init.uniform_(p.data)
                continue
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            elif init_kaiming:
                # not full supporting
                if init_normal:
                    nn.init.kaiming_normal_(p.data)
                else:
                    nn.init.kaiming_uniform_(p.data)
            else:
                if init_normal:
                    nn.init.normal_(p.data)
                else:
                    nn.init.uniform_(p.data)
