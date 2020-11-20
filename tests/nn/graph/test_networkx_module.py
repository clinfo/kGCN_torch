import torch
import torch.nn as nn

from kgcn_torch.nn.graph.networkx_module import NNGraph
from kgcn_torch.nn.graph.networkx_module import GraphModule
from kgcn_torch.nn.util_modules import Concatenate


def test_nngraph():
    pass

def test_graph_module_with_concat_graph():
    nngraph = GraphModule()
    nngraph.add_node('linear', nn.Linear(128, 64))
    nngraph.add_node('linear1', nn.Linear(128, 64))
    nngraph.add_node('concat', Concatenate(axis=1), previous=['linear', 'linear1'])        
    nngraph.add_node('linear2', nn.Linear(128, 32), previous=['concat'])    
    nngraph.add_node('out_linear', nn.Linear(32, 10), previous=['linear2'])
    x1 = torch.randn(16, 128)
    x2 = torch.randn(16, 128)    
    out = nngraph(x1, x2)
    assert list(out.shape) == [16, 10]


def test_graph_module_with_loop_sequence_graph():
    nngraph = GraphModule()
    nngraph.add_node('linear', nn.Linear(128, 64))
    nngraph.add_node('linear1', nn.Linear(64, 64), previous=['linear'])
    nngraph.add_node('linear2', nn.Linear(64, 32), previous=['linear1'])
    nngraph.add_node('linear3', nn.Linear(32, 32), previous=['linear2'])
    x = torch.randn(16, 128)
    out = nngraph(x)


    
