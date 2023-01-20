
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout

from torch_geometric.nn import GCNConv, global_sort_pool, SAGEConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.nn import MessagePassing, global_max_pool, global_add_pool



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


class DGCNN0(nn.Module):
    """
    Just an example of a PYG model, not really good though.

    UNUSED FOR NOW.
    """
    def __init__(self, num_features, num_classes):
        super(DGCNN0, self).__init__()

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


class DGCNN(nn.Module):
    """
    Uses fixed architecture
    """

    def __init__(self, dim_features, hidden_dim, dim_target):
        super(DGCNN, self).__init__()
        self.pyg_format = True
        self.k = 16 #self.ks[config.dataset.name][str(config['k'])]
        self.embedding_dim = hidden_dim
        self.num_layers = 4

        self.convs = []
        for layer in range(self.num_layers):
            input_dim = dim_features if layer == 0 else self.embedding_dim
            self.convs.append(DGCNNConv(input_dim, self.embedding_dim))
        self.total_latent_dim = self.num_layers * self.embedding_dim

        # Add last embedding
        self.convs.append(DGCNNConv(self.embedding_dim, 1))
        self.total_latent_dim += 1

        self.convs = nn.ModuleList(self.convs)

        # should we leave this fixed?
        self.conv1d_params1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        self.input_dense_dim = (dense_dim - 5 + 1) * 32

        self.hidden_dense_dim = 32 #config['dense_dim']
        self.dense_layer1 = nn.Sequential(nn.Linear(self.input_dense_dim, self.hidden_dense_dim),
                                         nn.ReLU())
        self.dense_layer2 = nn.Sequential( nn.Dropout(p=0.5),
                                         nn.Linear(self.hidden_dense_dim, dim_target))

    def forward(self, data):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = data.x, data.edge_index, data.batch

        hidden_repres = []

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            hidden_repres.append(x)

        # apply sortpool
        x_to_sortpool = torch.cat(hidden_repres, dim=1)
        x_1d = global_sort_pool(x_to_sortpool, batch, self.k)  # in the code the authors sort the last channel only

        # apply 1D convolutional layers
        x_1d = torch.unsqueeze(x_1d, dim=1)
        conv1d_res = F.relu(self.conv1d_params1(x_1d))
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
        conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1)

        # apply dense layer
        last_layer = self.dense_layer1(conv1d_res)
        out = self.dense_layer2(last_layer)

        return F.log_softmax(out, dim=1), out, last_layer


class DGCNNConv(MessagePassing):
    """
    From https://github.com/diningphil/gnn-comparison

    UNUSED FOR NOW.
    """

    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

        self.pyg_format = True

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]

        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphSAGE(nn.Module):

    def __init__(self, dim_features, dim_embedding, dim_target):
        super().__init__()
        self.pyg_format = True
        num_layers = 4
        self.aggregation = "max" # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        last_layer= F.relu(self.fc1(x))
        out = nn.Dropout(p=0.3)(last_layer)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1), out, last_layer



class GraphGAT(nn.Module):

    def __init__(self, nheads, dim_features, dim_embedding, dim_target):
        super().__init__()
        self.pyg_format = True
        
        self.conv = GATConv(dim_features, dim_embedding//2, nheads, edge_dim = 5)

        # For graph classification
        self.fc1 = nn.Linear(nheads * dim_embedding//2, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv(x, edge_index, edge_attr)

        x = global_add_pool(x, batch)

        last_layer= F.relu(self.fc1(x))
        out = nn.Dropout(p=0.2)(last_layer)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1), out, last_layer

