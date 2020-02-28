import torch.nn as nn
import torch_geometric.nn as geonn

from .base_module import BaseModule
from .utils import Initializer


class MultiedgeGCN(BaseModule):
    """This GCN Moudle is compatible with original kGCN GCN layer. Original one can treat multi state graph. These states are corresponding to each channels.

    ```

    ```
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 improved: bool=False,
                 cached: bool=False,
                 bias: bool=True,
                 normalize: bool=True,
                 adj_channel_num: int=1,
                 initializer: Initializer=Initializer.XAVIER_NORMAL,
                 **kwargs):
        super(MultiedgeGCN, self).__init__()
        self.output_dim = output_dim
        self.adj_channel_num = self.adj_channel_num
        self.gcns = nn.ModuleList([
            geonn.GCNConv(in_channels, out_channels,
                          improved, cached, bias, normalize, **kwargs)
            for i in range(adj_channel_num)
            ])
        self._initialize(initializer)

    def forward(self):
        pass
