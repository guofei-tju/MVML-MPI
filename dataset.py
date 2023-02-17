import pandas as pd
import torch
import dgl

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem

from utils import tokens_struct
from pubchemfp import GetPubChemFPs


class MolDataSet(Dataset):
    def __init__(self, data_path, labels_split=','):
        self.data = pd.read_csv(data_path)
        self.smiles = self.data['smiles'].to_list()
        self.labels = self.data['label'].to_list()
        self.labels_split = labels_split
        self.token = tokens_struct()
        self.node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

    def __getitem__(self, index):
        smiles = self.smiles[index]

        labels = self.labels[index]
        labels = [int(i) for i in labels.split(self.labels_split)]
        onehot = F.one_hot(torch.tensor(labels), 11).sum(dim=0)

        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_bigraph(mol, add_self_loop=True, node_featurizer=self.node_featurizer,
                               edge_featurizer=self.edge_featurizer)

        fp = []
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  # 167
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # 441
        fp_pubcfp = GetPubChemFPs(mol)  # 881
        fp_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp.extend(fp_pubcfp)
        fp.extend(fp_ecfp2)
        return self.token.encode(smiles), graph, fp, onehot

    def __len__(self):
        return len(self.smiles)


def collate(sample):
    encoded_smiles, graphs, fps, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    labels = torch.cat(labels).reshape(len(labels), -1).float()
    seq_len = [i.size for i in encoded_smiles]
    padded_smiles_batch = pad_sequence([torch.tensor(i) for i in encoded_smiles], batch_first=True)
    fps_t = torch.FloatTensor(fps)
    return {'smiles': padded_smiles_batch, 'seq_len': seq_len}, batched_graph, fps_t, labels
