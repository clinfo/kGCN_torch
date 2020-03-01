#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchex.nn as exnn
from torch_chemistry.datasets.tox21 import Tox21Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
import numpy as np
from sklearn.metrics import roc_auc_score


class GCN(nn.Module):
    def __init__(self, max_atom_types, output_channels):
        super(GCN, self).__init__()
        self.g1 = GCNConv(100, 42)
        self.g2 = GCNConv(42, 24)
        self.g3 = GCNConv(24, 16)
        self.l1 = nn.Linear(16, 10)
        self.l2 = exnn.Linear(output_channels)

    def forward(self, data):
        x = self.g1(data.x.float(), data.edge_index)
        x = F.relu(x)
        x = self.g2(x, data.edge_index)
        x = F.relu(x)
        x = self.g3(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.l1(x)
        x = self.l2(x)
        return torch.sigmoid(x)


def one_epoch(args, mode, model, device, loader, optimizer, epoch):
    if mode == "train":
        model.train()
    else:
        model.eval()
    total_loss = 0
    correct = 0
    n_valid_data = 0
    auc = 0
    loss_func = MaksedBCELoss()
    all_outputs = []
    all_labels = []
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if mode == "train":
            optimizer.zero_grad()
        labels = data.y.reshape(-1)
        mask = labels.ne(-1)
        output = model(data).reshape(-1)
        loss = loss_func(output, labels.float(), mask)
        total_loss += loss.item()
        all_labels += labels.masked_select(mask).reshape(-1).detach().cpu().numpy().tolist()
        all_outputs += output.masked_select(mask).reshape(-1).detach().cpu().numpy().tolist()
        correct += output.masked_select(labels.eq(1)).ge(0.5).sum()
        correct += output.masked_select(labels.eq(0)).le(0.5).sum()
        n_valid_data += mask.sum()
        if mode == "test":
            print(labels)
        if mode == "train":
            loss.backward()
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    mode,
                    epoch,
                    batch_idx * data.num_graphs,
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                )
            )

    print(
        "{} Epoch: {} {:.6f} AUC".format(
            mode, epoch, roc_auc_score(np.array(all_labels), np.array(all_outputs))
        )
    )

    total_loss /= n_valid_data

    print(
        "\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            mode, total_loss, correct, n_valid_data, 100.0 * correct / n_valid_data
        )
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch tox21 Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--max_atoms", type=int, default=150, metavar="N", help="set maximum atoms in dataset"
    )
    parser.add_argument(
        "--max_atom_types",
        type=int,
        default=100,
        metavar="N",
        help="set maximum number of the atom type in dataset",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    train_loader = DataLoader(Tox21Dataset("train"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Tox21Dataset("val"), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Tox21Dataset("test"), batch_size=args.batch_size, shuffle=True)

    model = GCN(args.max_atom_types, 12).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        one_epoch(args, "train", model, device, train_loader, optimizer, epoch)
        one_epoch(args, "val", model, device, val_loader, optimizer, epoch)
        scheduler.step()
    # one_epoch(args, 'test', model, device, test_loader, optimizer, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
