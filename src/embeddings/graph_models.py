
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout

from torch_geometric.nn import GCNConv, global_sort_pool, SAGEConv
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.aggr import SortAggregation



class GNN(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pyg_format = False

    def forward(self, x_in, adj, idx):
        # first message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer
        x = self.fc2(x)
        x = self.relu(torch.mm(adj, x))
        
        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)
        
        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        last_layer = self.relu(self.fc3(out))
        out = self.dropout(last_layer)
        out = self.fc4(out)

        return F.log_softmax(out, dim=1), out, last_layer


class DGCNN(nn.Module):
    """
    Just an example of a PYG model, not really good though
    """
    def __init__(self, num_features, num_classes):
        super(DGCNN, self).__init__()

        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 1)
        self.agg = SortAggregation(k=30)
        self.conv3 = Conv1d(1, 16, 97, 97)
        self.conv4 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(192, 128)
        self.drop_out = nn.Dropout(0.3)
        self.classifier_2 = Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.pyg_format = True

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)

        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_1 = self.drop_out(x_1)
        x_2 = torch.tanh(self.conv2(x_1, edge_index)) 
        x_2 = self.drop_out(x_2)
        x = torch.cat([x_1, x_2], dim=-1)
        x = self.agg(x, batch)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        last_layer = self.relu(self.classifier_1(x))
        out = self.drop_out(last_layer)
        out = self.classifier_2(out)
        classes = F.log_softmax(out, dim=-1)

        return classes, out, last_layer