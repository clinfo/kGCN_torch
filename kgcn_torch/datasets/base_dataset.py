from torch.utils.data import Dataset
import rdkit

from torch_geometric.data import InMemoryDataset


class InMemoryRdkitDataset(InMemoryDataset):
    edge_types = {rdkit.Chem.rdchem.BondType.SINGLE: 0,
                  rdkit.Chem.rdchem.BondType.DOUBLE: 1,
                  rdkit.Chem.rdchem.BondType.TRIPLE: 2,
                  rdkit.Chem.rdchem.BondType.AROMATIC: 3}

    def __init__(self, root, tranform=None, pre_transform=None, pre_filter=None):
        super(InMemoryRdkitDataset, self).__init__(root, tranform, pre_transform, pre_filter)

    def __str__(self):
        return self.__name__
