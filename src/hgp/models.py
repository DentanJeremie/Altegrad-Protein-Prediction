from src.utils.logging import logger
logger.warn('The code in this folder is a draft, untested and not integrated into the pipeline.')

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gat_conv import GATConv

from src.hgp.layers import GCN, HGPSLPool


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        #edge_attr = None
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

class BaselineGATModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.nhid
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.convs = [GATConv(self.num_features, self.hidden_size),
                      GATConv(self.hidden_size, self.hidden_size)]
        self.linear = torch.nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv in self.convs[:-1]: 
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index) 
        x = F.relu(x)
        x = gmp(x, batch)
        x = self.linear(x)

        return F.log_softmax(x, dim=-1)


# Basically the same as the baseline except we pass edge features 
class GATModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.nhid
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.n_edge_features = args.n_edge_features
        self.dropout = args.dropout_ratio
        self.convs = [GATConv(self.num_features, self.hidden_size, edge_dim = self.n_edge_features, dropout = self.dropout),
                      GATConv(self.hidden_size, self.hidden_size, edge_dim = self.n_edge_features, dropout = self.dropout)]
        self.linear = torch.nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        pos = data.pos
        #x = torch.cat([x, pos], dim = -1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        x = F.relu(x)

        x = gmp(x, batch)
        x = self.linear(x)

        return F.log_softmax(x, dim=-1)