import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import coalesce
from rdkit import Chem
from rdkit.Chem import rdmolops
import  pandas as pd

from ..base_dataset import InMemoryRdkitDataset
from ..utils import (
    check_download_file_size,
    check_local_file_size,
    extract_zipfile,
    get_mol_edge_index,
    download,
    protein_seq_to_vec,    
)
from ..utils import to_path


class MultiModalChemblDataset(InMemoryRdkitDataset):
    """ multi modal chembl dataset
    """
    _urls = {'url': 'https://github.com/clinfo/kGCN/files/5362776/CheMBL_MMP_table_data.zip',
             'filename': 'CheMBL_MMP_table_data.zip',
             'csvfilename': 'dataset_benchmark.tsv'}
    ONE_LETTER_AAs = 'XACDEFGHIKLMNPQRSTVWY'
    
    def __init__(
            self, savedir=".", max_n_types=100, smiles_canonical=True,
    ):
        self.filename = self._urls["filename"].replace(".zip", "")
        self.max_n_types = max_n_types
        self.smiles_canonical = smiles_canonical
        self.mols = None # set in def download(self).
        super(MultiModalChemblDataset, self).__init__(savedir, None, None, None)
        self._len = len(self.mols)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self._urls["filename"].replace(".zip", "")

    @property
    def processed_file_names(self):
        return self._urls["csvfilename"] + ".pt"

    def download(self):
        url = self._urls["url"]
        filename = self._urls["filename"]
        savefilename = to_path(self.root) / filename
        ## FIXME: download link is not work
        if savefilename.exists():
            local_file_size = check_local_file_size(savefilename)
            download_file_size = check_download_file_size(url)
            if local_file_size != download_file_size:
                download(url, filename, self.root)
        else:
            download(url, filename, self.root)
        extracted_files = extract_zipfile(savefilename, self.root)
        self.mols = self._get_valid_mols()
        return extracted_files

    def process(self):
        mols = self.mols
        data_list = []
        max_n_types = 0
        
        for data in mols:
            d = self._create_data_object(data, self.max_n_types)
            if d is not None:
                data_list.append(d)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _create_data_object(self, data, max_n_types):
        mol, protein_seq, label = data
        atoms = torch.tensor([m.GetAtomicNum() for m in mol.GetAtoms()])
        atoms = F.one_hot(atoms, max_n_types)
        edge_index, edge_attr = get_mol_edge_index(mol, self.edge_types)
        if edge_index.nelement() == 0:
            return None
        label = torch.LongTensor(label)
        n_atoms = mol.GetNumAtoms()
        edge_index, edge_attr = coalesce(edge_index, edge_attr, n_atoms, n_atoms)
        data = Data(atoms, edge_index, edge_attr, (protein_seq, label))
        return data

    def _create_element(self, data):
        label = torch.Tensor([data[1],]).long()
        mol = Chem.MolFromSmiles(data[4])
        if self.smiles_canonical:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(smiles)
        protein_seq = protein_seq_to_vec(data[5], 750)
        return mol, protein_seq, label
    
    def _get_valid_mols(self):
        mols = []
        df = self._read_tcv(self._urls["csvfilename"])
        for data in df.itertuples():
            mols.append(self._create_element(data))
        return mols

    def _read_tcv(self, filename, skiprows=None):
        df = pd.read_csv(filename, delimiter='\t')
        return df
    
    def __len__(self):
        return self._len

if __name__ == '__main__':
    from torch_geometric.data import DataLoader        
    d = MultiModalChemblDataset()
    train_loader = DataLoader(d, batch_size=16, shuffle=True)
    for d in train_loader:
        print(len(d.y))
