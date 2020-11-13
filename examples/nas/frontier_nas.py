#!/usr/bin/env python
import time
import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torchex.nn as exnn
import networkx as nx
from frontier_graph import NetworkxInterface

from inferno.extensions.layers.reshape import Concatenate
from inferno.extensions.containers import Graph

from thdbonas import Searcher, Trial


def conv2d(out_channels, kernel_size, stride):
    conv = nn.Sequential(
        exnn.Conv2d(out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride),
        nn.ReLU())
    return conv


class FlattenLinear(nn.Module):
    def __init__(self, out_channels):
        super(FlattenLinear, self).__init__()
        self.out_channels = out_channels
        self.linear = exnn.Linear(self.out_channels)
        self.flatten = exnn.Flatten()
    
    def forward(self, x):
        return F.relu(self.linear(self.flatten(x)))

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        self.cat = Concatenate()

    def reshape_1d(self, x):
        B = x.size(0)
        return x.view(B, -1)
        
    def forward(self, *inputs):
        reshaped_inputs = []
        for x in inputs:
            reshaped_inputs.append(self.reshape_1d(x))
        return self.cat(*reshaped_inputs)
    

class ModuleGen:
    def __init__(self):
        self.layer = []
        self.layer_dict = OrderedDict()
        self.layer_dict['concat'] = 0
        self._len = None

    def register(self, module_name: str, **params):
        self.layer.append((module_name, params))
        self.layer_dict[module_name] = 0
        for k in params:
            self.layer_dict[k] = 0

    def __getitem__(self, idx):
        module, vec = self.construct(idx)
        return module, vec

    def construct(self, idx):
        _layer_dict = copy.deepcopy(self.layer_dict)
        (module_name, params) = self.layer[idx]
        _layer_dict[module_name] = 1
        for k, v in params.items():
            _layer_dict[k] = v
        vec = list(_layer_dict.values())
        if module_name == 'conv2d':
            return conv2d(**params), vec
        elif module_name == 'linear':
            return FlattenLinear(**params), vec
        elif module_name == 'identity':
            return exnn.Flatten(), vec

    def get_linear(self, out_channels):
        _layer_dict = copy.deepcopy(self.layer_dict)
        _layer_dict['linear'] = 1
        _layer_dict['out_channels'] = out_channels
        vec = list(_layer_dict.values())
        return FlattenLinear(out_channels), vec

    def get_identity_vec(self):
        _layer_dict = copy.deepcopy(self.layer_dict)
        _layer_dict['identity'] = 1
        vec = list(_layer_dict.values())
        return vec

    def get_cat(self):
        _layer_dict = copy.deepcopy(self.layer_dict)
        _layer_dict['concat'] = 1
        vec = list(_layer_dict.values())
        return Concat(), vec

    def get_empty_mat(self, n_node: int):
        _layer_dict = copy.deepcopy(self.layer_dict)
        n_features = len(_layer_dict.values())
        mat = np.zeros((n_node, n_features))
        return mat

    def __len__(self):
        if self._len is None:
            self._len = len(self.layer)
        return self._len


class NetworkGeneratar:
    def __init__(self, graph, starts, ends, max_samples, dryrun_args):
        self.graph = graph
        self.starts = starts
        self.ends = ends
        self.dryrun_args = dryrun_args
        self.modulegen = ModuleGen()
        #self.modulegen.register('conv2d', out_channels=32, kernel_size=1, stride=1)
        self.modulegen.register('linear', out_channels=64)
        self.modulegen.register('identity')
        self.interface = NetworkxInterface(g)
        self.subgraph = self.get_subgraph(starts, ends, max_samples)
        self.n_subgraph = len(self.subgraph)
        self._len = None

    def get_subgraph(self, starts, ends, max_samples):
        return self.interface.sample(starts, ends, max_samples)

    def _construct_module(self, edge_list, _idx):
        module = Graph()
        # print(self.graph)
        for i in self.starts:
            vec = self.modulegen.get_identity_vec()
            module.add_input_node(f'{i}', vec=vec)
        node_dict = {}
        for (src, dst) in [list(self.graph.edges())[i-1] for i in edge_list]:
            src = int(src)
            dst = int(dst)
            if not dst in node_dict.keys():
                node_dict[dst] = [src]
            else:
                node_dict[dst].append(src)
        # print('-'*100)
        # print(self.graph.edges())
        # print([list(self.graph.edges())[i-1] for i in edge_list])
        # print(edge_list)
        # print(node_dict)
        for key, previous in sorted(node_dict.items(), key=lambda x: x[0]):
            layer_idx = _idx % len(self.modulegen)
            _idx //= len(self.modulegen)
            layer_names = [m._get_name() for m in module.modules()][1:] # skip 'Graph' module
            node_names = list(module.graph.nodes)
            # print('node_names', node_names)
            # print('key', key)
            # print('previous', previous)            
            if len(previous) == 1:
                mod, vec = self.modulegen[layer_idx]
                if mod._get_name() == 'Conv2d':
                    parents_indexes = [node_names.index(str(p)) for p in previous]
                    parents_module_names = [layer_names[i] for i in parents_indexes]
                    if 'FlattenLinear' in parents_module_names:
                        raise RuntimeError("can't append Conv2d after FlattenLinear.")
                # print(f'key => {key}', mod, [str(p) for p in previous], vec)
                # print(module.graph.nodes())
                # print(module.graph.edges())                
                module.add_node(f'{key}', mod, previous=[str(p) for p in previous], vec=vec)
            else:
                # print(key, previous)
                mod, vec = self.modulegen.get_cat()
                parents_indexes = [node_names.index(str(p)) for p in previous]
                parents_module_names = [layer_names[i] for i in parents_indexes]
                if 'FlattenLinear' in parents_module_names and 'Conv2d' in parents_module_names:
                    raise RuntimeError("can't concatinate FlattenLinear and Conv2d")
                module.add_node(f'{key}', mod, previous=[str(p) for p in previous], vec=vec)

        mod, vec = self.modulegen.get_linear(10)
        module.add_node(f'{int(key) + 1}', mod, vec=vec, previous=[f'{key}'])
        vec = self.modulegen.get_identity_vec()
        module.add_output_node(f'{int(key) + 2}', f'{int(key) + 1}', vec=vec)
        edges = [[int(e[0]) - 1, int(e[1]) - 1] for e in module.graph.edges()]
        node_features = self.modulegen.get_empty_mat(int(key) + 2)
        for node in module.graph.nodes(data=True):
            idx = int(node[0]) - 1
            node_features[idx, :] = node[1]['vec']
        y = module(*self.dryrun_args)
        return module, (edges, np.vstack(node_features))

    def __iter__(self):
        self.counter = 0
        return self

    def elongate(self, template_graph_module):
        template = template_graph_module.graph
        second_nodes = np.array(self.starts) + len(self.starts)
        dumy_graph, dumy_graph_edge_index_list = self._create_dumy_graph_for_elongation()
        self._len = None
        self.graph, edge_relation_with_subgraph, template_edges = self._update_graph(template, dumy_graph)
        self.subgraph = self.update_subgraph(template,
                                             dumy_graph,
                                             dumy_graph_edge_index_list,                                             
                                             self.graph,
                                             edge_relation_with_subgraph,
                                             template_edges)
        self.n_subgraph = len(self.subgraph)
        return self

    def update_subgraph(self, template, dumy_graph, subgraph_edge_index_lists, graph, relation_subgraph_to_original, template_edges):
        relation_original_to_subgraph = {k: v for (k, v) in relation_subgraph_to_original.items()}
        subgraph_edges = list(dumy_graph.edges())
        edges_to_edge_index = {e: idx + 1 for idx, e in enumerate(template.edges())}
        _base_edge_index = [edges_to_edge_index[e] for e in template_edges]

        subgraph = []
        # print(relation_subgraph_to_original)
        for edge_index_list in subgraph_edge_index_lists:
            edge_indices = []
            for e in edge_index_list:
                if subgraph_edges[e-1] in relation_subgraph_to_original.keys():
                    sub_e = relation_subgraph_to_original[subgraph_edges[e-1]]
                    # print(sub_e)
                    edge_indices.append(edges_to_edge_index[sub_e])
            edge_index = _base_edge_index + edge_indices
            subgraph.append(edge_index)
        return subgraph
    
    def _update_graph(self, template_graph, graph):
        removed_nodes = list(template_graph.nodes)[-3:]
        remove_nodes_list = []
        max_elongation_nodes = 100
        for n in removed_nodes:
            template_graph.remove_node(n)

        template_edges = list(template_graph.edges())
        last_node = list(template_graph.nodes)[-1]
        new_edges = []
        
        new_nodes = [f'{i + int(last_node)}' for i in range(1, 4)]

        ## add new nodes and edges        
        for n in new_nodes:
            template_graph.add_edge(last_node, n)

        edge_relation_between_subgraph_and_original = {
            (3, 4): (last_node, new_nodes[0]),
            (3, 5): (last_node, new_nodes[1]),
            (3, 6): (last_node, new_nodes[2]),                            
            }
            
        edge_relation_between_subgraph_and_original[(4, 6)] = (new_nodes[0], new_nodes[-1])
        edge_relation_between_subgraph_and_original[(5, 6)] = (new_nodes[1], new_nodes[-1])        
        template_graph.add_edge(new_nodes[0], new_nodes[-1])
        template_graph.add_edge(new_nodes[1], new_nodes[-1])
        # right
        for step in range(3, max_elongation_nodes, 3):
            if f'{int(new_nodes[0]) - step}' in list(template_graph.nodes):
                template_graph.add_edge(f'{int(new_nodes[0]) - step}', new_nodes[0])
                edge_relation_between_subgraph_and_original[(2, 4)] = (f'{int(new_nodes[0]) - step}', new_nodes[0])
                break

        # left
        for step in range(3, max_elongation_nodes, 3):            
            if f'{int(new_nodes[-2]) - step}' in list(template_graph.nodes):
                template_graph.add_edge(f'{int(new_nodes[-2]) - step}', new_nodes[1])
                edge_relation_between_subgraph_and_original[(2, 5)] = (f'{int(new_nodes[-2]) - step}', new_nodes[1])
                break
        template_graph.add_edge(new_nodes[-1], f'{int(new_nodes[-1]) + 1}')
        return template_graph, edge_relation_between_subgraph_and_original, template_edges

    def _create_dumy_graph_for_elongation(self):
        _max_graph_samples = 100
        g = nx.DiGraph()
        starts = [1,]
        ends = [7,]
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(2, 5)                
        g.add_edge(3, 4)
        g.add_edge(3, 5)
        g.add_edge(3, 6)
        g.add_edge(4, 6)
        g.add_edge(5, 6)        
        g.add_edge(6, 7)
        ns = NetworkxInterface(g)
        graphs = ns.sample(starts, ends, _max_graph_samples)
        edge_list = self.edge_index_to_edge_nodes(graphs[0], g.edges())
        for idx, edge_index_list in enumerate(graphs):
            edge_list = self.edge_index_to_edge_nodes(edge_index_list, g.edges())            
            draw_edge_list(edge_list, f'subgraph{idx:03d}.png')
        return g, graphs

    def edge_index_to_edge_nodes(self, edge_index_list, edges):
        return [list(edges)[edge_index-1] for edge_index in edge_index_list]

    def _draw_dumy_graph(self, g, filename):
        pos_dir = {
            '1': np.array([0, 0.25]),
            '2': np.array([0, 0]),
            '3': np.array([-0.25, -0.25]),
            '4': np.array([0.25, -0.25]),
            '5': np.array([0, -0.5]),
            '6': np.array([-0.25, -0.75]),
            '7': np.array([0.25, -0.75]),
            '8': np.array([0, -1]),
            '9': np.array([0, -1.25]),
        }
        pos = nx.spring_layout(g)        
        for k in pos_dir:
            pos[int(k)] = pos_dir[k]
        nx.draw(g, pos)
        plt.savefig(filename)
        plt.clf()

    def __next__(self):
        if self.counter <= len(self):
            self.counter += 1
            while True:
                try:
                    module, edges, node_features = self[self.counter]
                    module(*self.dryrun_args)
                    break
                except RuntimeError as e:
                    self.counter += 1
                    pass
            return module, edges, node_features
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        _idx = idx
        subgraph_idx = _idx % self.n_subgraph
        edge_list = self.subgraph[subgraph_idx]
        _idx //= self.n_subgraph
        module = self._construct_module(edge_list, _idx)
        return module

    def __len__(self):
        if self._len is None:
            n_layer = len(self.modulegen)
            n = 0
            for graph in self.subgraph:
                n_edges = len(graph) # return number of edges
                n += n_layer ** n_edges
            self._len = n
        return self._len


def objectve(trial):
    model, _ = trial.graph
    use_cuda = True

    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    lr = 0.01
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    correct = 0
    s = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        d1, d2 = torch.split(data, 14, dim=2)
        optimizer.zero_grad()
        output = model(d1, d2)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        total_loss += loss.detach().cpu().item()
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx > 30:
            break
    print(f"time = {time.time() - s}")
    acc = 100. * correct / ((batch_idx + 1) * batch_size)
    print(acc)
    return acc


def draw_module(model, filename='test.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-2, 1])
    ax.set_aspect('equal')
    pos_dir = {
        '1': np.array([-0.25, 0.25]),
        '2': np.array([0.25, 0.25]),
        '3': np.array([-0.25, 0]),
        '4': np.array([0.25, 0]),
        '5': np.array([0, -0.25]),
        '6': np.array([-0.25, -0.50]),
        '7': np.array([0.25, -0.50]),
        '8': np.array([0, -0.75]),
        '9': np.array([0, -1.0]),
        '10': np.array([0, -1.25]),
        '11': np.array([0, -1.5]),                        
        }
    try:
        g = model.graph
        print(g.nodes())        
    except:
        g = model
    pos = nx.spring_layout(g)
    for k in pos_dir:
        pos[k] = pos_dir[k]
    pos = {n: pos_dir[str(n)] for n in g.nodes()}
    print(pos)
    nx.draw(g, pos, ax)
    plt.savefig(filename)
    plt.clf()

    
def elongation(model):
    g = model.graph
    remove_more_than = len(g) - 2
    remove_list = []
    for n in g.nodes():
        if int(n) >= remove_more_than:
            remove_list.append(n)
    for n in remove_list:
        g.remove_node(n)
    draw_module(g)

def draw_edge_list(edges, filename='subgraph.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-2, 1])
    ax.set_aspect('equal')
    pos_dir = {
        '1': np.array([0, 0.5]),
        '2': np.array([0, 0.25]),
        '3': np.array([0, 0]),        
        '4': np.array([-0.25, -0.25]),
        '5': np.array([0.25, -0.25]),
        '6': np.array([0, -0.50]),
        '7': np.array([0, -0.75]),
        }
    g = nx.DiGraph()    
    for e in edges:
        g.add_edge(e[0], e[1])
    labels = {n: n for n in g.nodes()}        
    pos = nx.spring_layout(g)
    for k in pos_dir:
        pos[k] = pos_dir[k]
    pos = {n: pos_dir[str(n)] for n in g.nodes()}
    nx.draw(g, pos, ax=ax, labels=labels)
    plt.savefig(filename)
    plt.clf()

    
if __name__ == "__main__":
    g = nx.DiGraph()
    starts = [1, 2]
    ends = [9,]
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 5)
    g.add_edge(3, 6)
    g.add_edge(4, 5)
    g.add_edge(4, 7)
    g.add_edge(5, 6)
    g.add_edge(5, 7)
    g.add_edge(5, 8)
    g.add_edge(6, 8)
    g.add_edge(7, 8)
    g.add_edge(8, 9)    
    sample_size = 100
    ns = NetworkxInterface(g)
    graphs = ns.sample(starts, ends, 100)
    edges = list(g.edges())
    x = torch.rand(128, 392)
    # 392 + 392 # for linear layer
    ng = NetworkGeneratar(g, starts, ends, 100, dryrun_args=(x, x))
    print(len(ng))
    models = []
    total_num = 10
    #draw_module(ng[98][0])
    # ng = ng.elongate(ng[0][0])
    # for i in range(100):
    #     print(ng[i])
    ##
    num_node_features = 4
    for i in range(10):
        searcher = Searcher()
        print('size of ng', len(ng))
        samples = np.random.randint(0, len(ng), size=sample_size)        
        searcher.register_trial('graph', [ng[i] for i in samples])
        n_trials = 40
        n_random_trials = 10
        model_kwargs = dict(
            input_dim=num_node_features,
            n_train_epochs=400,
        )
        result = searcher.search(objectve,
                                 n_trials=n_trials,
                                 deep_surrogate_model=f'thdbonas.deep_surrogate_models:GCNSurrogateModel',
                                 n_random_trials=n_random_trials,
                                 model_kwargs=model_kwargs)
        print(f'{i} trial', result.max_value_idx, result.best_trial, result.best_value)
        ng = ng.elongate(ng[result.max_value_idx][0])

