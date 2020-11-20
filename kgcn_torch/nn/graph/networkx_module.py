from collections import OrderedDict

import networkx as nx
import torch.nn as nn


class NNGraph(nx.DiGraph):
    def __init__(self, *inputs, **inputs_dict):
        super(NNGraph, self).__init__(*inputs, **inputs_dict)


class GraphModule(nn.Module):
    ## inferno like interface
    ## only support single thread
    def __init__(self):
        super(GraphModule, self).__init__()
        self._graph = NNGraph()
        self._module_dict = nn.ModuleDict()
        self._first_forward_flag = False
        self._input_nodes = []

    def _check_valid_graph(self):
        pass

    @property
    def input_nodes(self):
        return self._input_nodes
    
    @property
    def output_nodes(self):
        pass

    def add_input_node(self, name, module=None, **attr):
        if not self._first_forward_flag:
            self._graph.add_node(name, **attr)
            self._module_dict[name] = module
        else:
            pass
    
    def add_output_node(self, name, module, **attr):
        if not self._first_forward_flag:
            self._graph.add_node(name, **attr)
            self._module_dict[name] = module
        else:
            pass

    def add_node(self, name, module, previous: list or None=None, **attr):
        if not self._first_forward_flag:
            self._graph.add_node(name, **attr)
            if previous is not None:
                if not isinstance(previous, list):
                    raise TypeError(f'type of previous is list. previous = {previous}, '
                                    f'type(previous) = {type(previous)}')
                for p in previous:
                    self._graph.add_edge(p, name)
            else:
                self._input_nodes.append(name)
            self._module_dict[name] = module
        else:
            raise Exception('error')

    def _forward(self, *inputs):
        assert len(inputs) == len(self._input_nodes), (f'number of inputs error. expected '
                                                       f'{len(self._input_nodes)}')
        second_layer = []
        topo_sort_nodes = list(nx.topological_sort(self._graph))
        outputs = OrderedDict()
        for module_name, x in zip(self._input_nodes, inputs):
            o = self._module_dict[module_name] (x)
            outputs[module_name] = o
            topo_sort_nodes.remove(module_name)
        for n in topo_sort_nodes:
            previous = list(self._graph.predecessors(n))
            _inputs = [outputs[p] for p in previous]
            outputs[n] = self._module_dict[n] (*_inputs)
            for p in previous:
                del outputs[p]
        out = list(outputs.values())
        if len(out) == 1:
            return out[0]
        else:
            return out
            
    def forward(self, *inputs):
        self._check_valid_graph()
        out = self._forward(*inputs)
        return out
