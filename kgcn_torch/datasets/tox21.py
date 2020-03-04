import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import coalesce
from rdkit import Chem
from rdkit.Chem import rdmolops

from .base_dataset import InMemoryRdkitDataset
from .utils import (
    check_download_file_size,
    check_local_file_size,
    extract_zipfile,
    get_mol_edge_index,
    download,
)
from ..utils import to_path


class Tox21Dataset(InMemoryRdkitDataset):
    """ tox21 dataset
    """
    _urls = {
        "train": {
            "url": "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf",
            "filename": "tox21_10k_data_all.sdf.zip",
        },
        "val": {
            "url": "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf",
            "filename": "tox21_10k_challenge_test.sdf.zip",
        },
        "test": {
            "url": "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresdf",
            "filename": "tox21_10k_challenge_score.sdf.zip",
        },
    }
    _label_names = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    def __init__(
            self, target="train", savedir=".", none_label=-1, max_n_atoms=150, max_n_types=100
    ):
        self.target = target
        self.filename = self._urls[self.target]["filename"].replace(".zip", "")
        self.none_label = none_label
        self.max_n_atoms = max_n_atoms
        self.max_n_types = max_n_types
        self.mol = None # set in def download(self).
        super(Tox21Dataset, self).__init__(savedir, None, None, None)
        self._len = len(self.mol)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self._urls[self.target]["filename"].replace(".zip", "")

    @property
    def processed_file_names(self):
        return self._urls[self.target]["filename"].replace(".sdf.zip", "") + ".pt"

    def download(self):
        url = self._urls[self.target]["url"]
        filename = self._urls[self.target]["filename"]
        savefilename = to_path(self.root) / filename
        if savefilename.exists():
            local_file_size = check_local_file_size(savefilename)
            download_file_size = check_download_file_size(url)
            if local_file_size != download_file_size:
                download(url, filename, self.root)
        else:
            download(url, filename, self.root)
        extracted_files = extract_zipfile(savefilename, self.root)
        self.mol = self._get_valid_mols()
        return extracted_files

    def process(self):
        mols = self.mol
        data_list = []
        max_n_types = 0
        for mol in mols:
            n_types = torch.max(torch.tensor([m.GetAtomicNum() for m in mol.GetAtoms()]))
            if n_types > max_n_types:
                max_n_types = n_types
        max_n_types = self.max_n_types
        for m in mols:
            d = self._create_data_object(m, max_n_types)
            if d is not None:
                data_list.append(d)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _create_data_object(self, mol, max_n_types):
        atoms = torch.tensor([m.GetAtomicNum() for m in mol.GetAtoms()])
        atoms = F.one_hot(atoms, max_n_types)
        edge_index, edge_attr = get_mol_edge_index(mol, self.edge_types)
        if edge_index.nelement() == 0:
            return None
        label = self._get_label(mol).long()
        n_atoms = mol.GetNumAtoms()
        edge_index, edge_attr = coalesce(edge_index, edge_attr, n_atoms, n_atoms)
        data = Data(atoms, edge_index, edge_attr, label)
        # data.num_nodes = self.max_n_atoms
        return data

    def _get_valid_mols(self):
        tmpmols = Chem.SDMolSupplier(self.filename)
        mols = []
        for m in tmpmols:
            if m is None:
                continue
            try:
                rdmolops.GetAdjacencyMatrix(m)
            except Exception as e:
                print(e)
                continue
            edge_index, _ = get_mol_edge_index(m, self.edge_types)
            if edge_index.nelement() == 0:
                continue
            mols.append(m)
        return mols

    def _get_label(self, mol: Chem):
        labels = []
        for label in self._label_names:
            if mol.HasProp(label):
                labels.append(int(mol.GetProp(label)))
            else:
                labels.append(self.none_label)
        return torch.tensor(labels)

    def __len__(self):
        return self._len
