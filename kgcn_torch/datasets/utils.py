import os
import shutil
from pathlib import Path
from zipfile import ZipFile
from typing import List

import requests
import torch
import torch.nn.functional as F

from ..utils import to_Path


def check_download_file_size(url: str) -> int:
    res = requests.head(url)
    size = res.headers['content-length']
    return int(size)

def check_local_file_size(filename: str) -> int:
    p = to_Path(filename)
    info = os.stat(p)
    return info.st_size

def download(url: str = '', filename: str = '', savedir: str ='.') -> int:
    savefile = to_Path(savedir) / filename
    if not savefile.exists():
        with requests.get(url, stream=True) as r:
            with open(savefile, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    return savefile

def extract_zipfile(zfilename: str, extractdir: str ='.') -> List[str]:
    with ZipFile(zfilename) as zipfile:
        zipfile.extractall(extractdir)
        namelist = zipfile.namelist()
    return namelist

def to_sparse(x: torch.tensor, max_size: int = None):
    """ ref: https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809 """
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)

    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    if max_size is None:
        return sparse_tensortype(indices, values, x.size())
    else:
        return sparse_tensortype(indices, values, (max_size, max_size))

def get_mol_edge_index(mol, edge_types: dict):
    row, col, bond_idx = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_idx += 2 * [edge_types[bond.GetBondType()]]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = F.one_hot(torch.tensor(bond_idx).long(),
                          num_classes=len(edge_types)).to(torch.long)
    return edge_index, edge_attr

def to_one_hot(x: torch.tensor, n_classes: int):
    length = len(x)
    if length < n_classes:
        _x = torch.arange(length)
        out = torch.zeros(length, n_classes)
        out[_x, x] = 1
        return out
    return torch.eye(length, n_classes)[x, :]
