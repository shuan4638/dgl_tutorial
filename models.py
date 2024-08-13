import torch
import torch.nn as nn

import dgl
import dgllife
from dgllife.model import MPNNGNN, WeightedSumAndMax    

class MyGNN(nn.Module):
    def __init__(self,
                 node_in_feats=74,
                 edge_in_feats=12,
                 node_hidden_dim=64,
                 edge_hidden_feats=32,
                 num_step_message_passing=3,
                 n_classes = 100):
        
        super(MyGNN, self).__init__()
                
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_hidden_dim,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.readout = WeightedSumAndMax(node_hidden_dim)
        
        self.ff = nn.Linear(node_hidden_dim*2, n_classes)
        
    def forward(self, graph):
        # get features from graph
        node_feats = graph.ndata['h']
        edge_feats = graph.edata['e']
        # message passing
        node_feats = self.mpnn(graph, node_feats, edge_feats)
        # readout
        readout = self.readout(graph, node_feats)
        # linear mlp
        output = self.ff(readout)
        return output