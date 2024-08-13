import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rdkit import Chem
import dgl
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

def SmilesToGraph(smi):
    mol = Chem.MolFromSmiles(smi)
    node_featurizer = CanonicalAtomFeaturizer() 
    edge_featurizer = CanonicalBondFeaturizer()
    graph = mol_to_bigraph(mol, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer)
    return graph


class DGLDataset(object):
    def __init__(self, data_path='odor_dataset.csv', split='train'):
        df = pd.read_csv(data_path)
        self.X = [smi for smi, spl in zip(df['SMILES'], df['Split']) if spl == split]
        self.Y = [smi for smi, spl in zip(df['Class'], df['Split']) if spl == split]

    def __getitem__(self, item):
        smi, label = self.X[item], self.Y[item]
        graph = SmilesToGraph(smi)
        return graph, label

    def __len__(self):
        return len(self.X)
    
    
def collate_molgraphs(data):
    graphs, labels = map(list, zip(*data))
    batch_graph = dgl.batch(graphs)
    return batch_graph, torch.LongTensor(labels)

def get_dataloaders(data_path='odor_dataset.csv', batch_size=16):
    train_set = DGLDataset(split='train')
    test_set = DGLDataset(split='test')
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, collate_fn=collate_molgraphs, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, collate_fn=collate_molgraphs)
    return train_loader, test_loader