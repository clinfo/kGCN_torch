import rdkit

from torch_geometric.data import InMemoryDataset


class InMemoryRdkitDataset(InMemoryDataset):
    """ Abstract database class for pytorch_geometirc models.
    """
    edge_types = {
        rdkit.Chem.rdchem.BondType.SINGLE: 0,
        rdkit.Chem.rdchem.BondType.DOUBLE: 1,
        rdkit.Chem.rdchem.BondType.TRIPLE: 2,
        rdkit.Chem.rdchem.BondType.AROMATIC: 3,
    }

    def __init__(self, root, tranform=None, pre_transform=None, pre_filter=None):
        super(InMemoryRdkitDataset, self).__init__(root, tranform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        raise NotImplementedError("")

    @property
    def processed_file_names(self):
        raise NotImplementedError("")

    def download(self):
        raise NotImplementedError("")

    def __str__(self):
        return self.__class__.__name__
