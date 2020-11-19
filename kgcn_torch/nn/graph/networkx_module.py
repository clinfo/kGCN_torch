import networkx as nx
import torch.nn as nn


class BaseGraphModule(nn.Module, nx.DiGraph):
    def __init__(self):
        super(BaseGraphModule, self).__init__()



class GraphModule(BaseGraphModule):
    def __init__(self):
        super(GraphModule, self).__init__()        

    def forward(self, x):
        pass


