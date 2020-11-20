import time
import warnings

import torch
import torch.nn as nn
import pytest

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

@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_to_gpu():
    nngraph = GraphModule()
    nngraph.add_node('linear', nn.Linear(128, 64))
    nngraph.add_node('linear1', nn.Linear(64, 64), previous=['linear'])
    nngraph.add_node('linear2', nn.Linear(64, 32), previous=['linear1'])
    nngraph.add_node('linear3', nn.Linear(32, 32), previous=['linear2'])
    nngraph.to('cuda')
    x = torch.randn(16, 128).to('cuda')
    out = nngraph(x)
    assert next(nngraph.parameters()).is_cuda
    
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_mnist_performance():
    import torchex.nn as exnn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    device = torch.device("cuda")
    nngraph = GraphModule()
    f = exnn.Flatten()
    nngraph.add_node('flatten', exnn.Flatten())    
    nngraph.add_node('linear', nn.Linear(784, 128), previous=['flatten'])
    nngraph.add_node('relu', nn.ReLU(), previous=['linear'])    
    nngraph.add_node('linear1', nn.Linear(128, 64), previous=['relu'])
    nngraph.add_node('relu1', nn.ReLU(), previous=['linear1'])    
    nngraph.add_node('linear2', nn.Linear(64, 10), previous=['relu1'])

    model = nngraph
    model.to(device)
    lr = 0.001
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(2):
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            s = time.time()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            output = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            total_loss += loss.detach().cpu().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            e = time.time()
        acc = 100. * correct / ((batch_idx + 1) * batch_size)
    warnings.warn(f'mnist train acc = {acc}, one iteration time = {e - s} s')
    assert acc > 90.
    


    
