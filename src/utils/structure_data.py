import typing as t

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch

from src.utils.pathtools import project
from src.utils.logging import logger


NODE_FEATURES_ENCODED_COLUMNS = list(range(3, 25))
NODE_FEATURES_TO_SCALE_COLUMNS = list(range(0,3)) + list(range(25, 86))
EDGE_FEATURES_ENCODED_COLUMNS = list(range(1, 5))
EDGE_FEATURES_TO_SCALE_COLUMNS = [0]


class StructureData():
    def __init__(self):
        # Standard format
        self._adjacency_matrixes = None
        self._node_features = None
        self._edge_features = None
        # Format for HGP
        self._adjacency_matrixes_hgp = None
        self._node_features_hgp = None
        self._edge_features_hgp = None
        self._edge_index_hgp = None
        # PCA reductions
        self._reduced_node_features = None
        self._reduced_edge_features = None
        self._reduced_node_features_hgp = None
        self._reduced_edge_features_hgp = None

# ------------------ PROPERTIES ------------------

    @property
    def adjacency_matrixes(self):
        if self._adjacency_matrixes is None:
            self.build_structure_data()
        return self._adjacency_matrixes

    @property
    def node_features(self):
        if self._node_features is None:
            self.build_structure_data()
        return self._node_features

    @property
    def edge_features(self):
        if self._edge_features is None:
            self.build_structure_data()
        return self._edge_features

    @property
    def adjacency_matrixes_hgp(self):
        if self._adjacency_matrixes_hgp is None:
            self.build_structure_data_for_hgp()
        return self._adjacency_matrixes_hgp

    @property
    def node_features_hgp(self):
        if self._node_features_hgp is None:
            self.build_structure_data_for_hgp()
        return self._node_features_hgp

    @property
    def edge_features_hgp(self):
        if self._edge_features_hgp is None:
            self.build_structure_data_for_hgp()
        return self._edge_features_hgp

    @property
    def edge_index_hgp(self):
        if self._edge_index_hgp is None:
            self.build_structure_data_for_hgp()
        return self._edge_index_hgp

    @property
    def reduced_node_features(self):
        if self._reduced_node_features is None:
            self.build_structure_data(pca_reduction=True)
        return self._reduced_node_features

    @property
    def reduced_edge_features(self):
        if self._reduced_edge_features is None:
            self.build_structure_data(pca_reduction=True)
        return self._reduced_edge_features

    @property
    def reduced_node_features_hgp(self):
        if self._reduced_node_features_hgp is None:
            self.build_structure_data_for_hgp(pca_reduction=True)
        return self._reduced_node_features_hgp

    @property
    def reduced_edge_features_hgp(self):
        if self._reduced_edge_features_hgp is None:
            self.build_structure_data_for_hgp(pca_reduction=True)
        return self._reduced_edge_features_hgp

# ------------------ RAW DATA STRUCTURE ------------------

    def build_structure_data(self, pca_reduction: bool = False) -> t.Tuple[
        t.List[sp.csr_matrix],
        t.List[np.ndarray],
        t.List[np.ndarray],
        ]:
        """Get the structure data of the protein graphs. This function builds:
        * A list of adjacency matrixes called `adjacency_matrixes`.
        The k-th element of this list is a `scipy.sparse.csr_matrix` representing the
        adjacency matrix of the k-th protein graph in the data.
        * A list of node features called `node_features`.
        The k-th element of this list is a `numpy.ndarray` representing the node features
        of all the nodes of this graph
        * A list of edge features called `node_features`.
        The k-th element of this list is a `numpy.ndarray` representing the edge features
        of all the edges of this graph
        
        :param pca_reduction: A boolean indicating if the features must be reduced with PCA
        :returns: The tuple  `(adjacency_matrixes, node_features, edge_features)`
        """
        if pca_reduction:
            logger.info('Building raw structure data reduced with PCA...')
        else:
            logger.info('Building raw structure data...')

        graph_indicator = np.loadtxt(project.graph_indicator, dtype=np.int64)
        _, graph_size = np.unique(graph_indicator, return_counts=True)
        edges = np.loadtxt(project.edgelist, dtype=np.int64, delimiter=",")
        raw_adjacency = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
        raw_adjacency += raw_adjacency.T

        node_attr = np.loadtxt(project.node_attributes, delimiter=",")
        edge_attr = np.loadtxt(project.edge_attributes, delimiter=",")

        if pca_reduction:
            node_attr, edge_attr = self.reduce_with_pca(node_attr, edge_attr)

        adjacency_matrixes: t.List[sp.csr_matrix] = []
        node_features: t.List[np.ndarray] = []
        edge_features: t.List[np.ndarray] = []
        idx_n = 0
        idx_m = 0
        for i in range(graph_size.size):
            adjacency_matrixes.append(raw_adjacency[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
            edge_features.append(edge_attr[idx_m:idx_m+adjacency_matrixes[i].nnz,:])
            node_features.append(node_attr[idx_n:idx_n+graph_size[i],:])
            idx_n += graph_size[i]
            idx_m += adjacency_matrixes[i].nnz

        # Storing the results
        if not pca_reduction:
            (
                self._adjacency_matrixes,
                self._node_features,
                self._edge_features
            ) = adjacency_matrixes, node_features, edge_features
        else:
            (
                self._adjacency_matrixes,
                self._reduced_node_features,
                self._reduced_edge_features
            ) = adjacency_matrixes, node_features, edge_features

        self.normalize_adjacency(self._adjacency_matrixes)

    def build_structure_data_for_hgp(self, pca_reduction: bool = False) -> t.Tuple[
        t.List[sp.csr_matrix],
        t.List[np.ndarray],
        t.List[np.ndarray],
        t.List[torch.Tensor]
        ]:
        """Get the structure data of the protein graphs. 
        
        This function builds:
        * A list of adjacency matrixes called `adjacency_matrixes`.
        The k-th element of this list is a `scipy.sparse.csr_matrix` representing the
        adjacency matrix of the k-th protein graph in the data.
        * A list of node features called `node_features`.
        The k-th element of this list is a `numpy.ndarray` representing the node features
        of all the nodes of this graph
        * A list of edge features called `node_features`.
        The k-th element of this list is a `numpy.ndarray` representing the edge features
        of all the edges of this graph
        * A list of edges indes called 'edge_index`.
        The k-th element of this list is a `torch.Tensor` of size (2, n) where `n`is the 
        number of edges in the k-th graph. The tuples formed by the first and second lines
        are the indexes of the edges of the k-th graph.
        
        :param pca_reduction: A boolean indicating if the features must be reduced with PCA
        :returns: The tuple  `(adjacency_matrixes, node_features, edge_features, edge_index)`
        """
        if pca_reduction:
            logger.info('Building raw structure data for HGP reduced with PCA...')
        else:
            logger.info('Building raw structure data for HGP...')

        graph_indicator = np.loadtxt(project.graph_indicator, dtype=np.int64)
        _, graph_size = np.unique(graph_indicator, return_counts=True)
        edges = np.loadtxt(project.edgelist, dtype=np.int64, delimiter=",")
        
        # Formatting edges for HGP
        # First, adding backward edges
        edges_inv = np.vstack((edges[:,1], edges[:,0]))
        edges = np.vstack((edges, edges_inv.T))
        # Then, sorting the edges
        s = edges[:,0]*graph_indicator.size + edges[:,1]
        idx_sort = np.argsort(s)
        edges = edges[idx_sort,:]
        # Finally, building the matrix
        edges, idx_unique =  np.unique(edges, axis=0, return_index=True)
        raw_adjacency = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))

        # Loading the attributes and re-aranging the edges attributes
        node_attr = np.loadtxt(project.node_attributes, delimiter=",")
        edge_attr = np.loadtxt(project.edge_attributes, delimiter=",")
        edge_attr = np.vstack((edge_attr,edge_attr))
        edge_attr = edge_attr[idx_sort,:]
        edge_attr = edge_attr[idx_unique,:]

        # PCA
        if pca_reduction:
            node_attr, edge_attr = self.reduce_with_pca(node_attr, edge_attr)

        # Building the output
        adjacency_matrixes: t.List[sp.csr_matrix] = []
        node_features: t.List[np.ndarray] = []
        edge_features: t.List[np.ndarray] = []
        edge_index: t.List[torch.Tensor] = []
        idx_n = 0
        idx_m = 0
        for i in range(graph_size.size):
            adjacency_matrixes.append(raw_adjacency[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
            edge_features.append(edge_attr[idx_m:idx_m+adjacency_matrixes[i].nnz,:])
            node_features.append(node_attr[idx_n:idx_n+graph_size[i],:])
            edge_index_raw = torch.from_numpy(np.array([
                edges[idx_m:idx_m+adjacency_matrixes[i].nnz][:,0], 
                edges[idx_m:idx_m+adjacency_matrixes[i].nnz][:,1]]))
            edge_index.append(edge_index_raw - edge_index_raw.min())
            idx_n += graph_size[i]
            idx_m += adjacency_matrixes[i].nnz

        # Storing the results
        if not pca_reduction:
            (
                self._adjacency_matrixes_hgp,
                self._node_features_hgp,
                self._edge_features_hgp,
                self._edge_index_hgp,
            ) = adjacency_matrixes, node_features, edge_features, edge_index
        else:
            (
                self._adjacency_matrixes_hgp,
                self._reduced_node_features_hgp,
                self._reduced_edge_features_hgp,
                self._edge_index_hgp,
            ) = adjacency_matrixes, node_features, edge_features, edge_index

        self.normalize_adjacency(self._adjacency_matrixes_hgp)

# ------------------ PCA ------------------

    def reduce_with_pca(self, node_features_raw: np.ndarray, edge_features_raw: np.ndarray) -> t.Tuple[np.ndarray]:
        """Reduces the node and edge attributes of the structure data with PCA.

        :param node_features_raw: An array representing the node features
        :param edge_features_raw: An array representing the edge features
        :returns: The arrays after PCA: `node_features_reduced`, `edge_features_reduced`
        """

        logger.debug("Reducing the node features...")
        node_features_df = pd.DataFrame(node_features_raw)
        # Scaling
        scaler = StandardScaler()
        scaled_columns  = scaler.fit_transform(node_features_df[NODE_FEATURES_TO_SCALE_COLUMNS]) 
        encoded_columns = node_features_df[NODE_FEATURES_ENCODED_COLUMNS]
        node_features_scaled = np.concatenate([scaled_columns, encoded_columns], axis=1)
        # PCA
        pca = PCA(n_components=0.95)
        pca.fit(node_features_scaled)
        node_features_reduced = pca.transform(node_features_scaled)

        logger.debug("Reducing the edge features...")
        edge_features_df = pd.DataFrame(edge_features_raw)
        # Scaling
        scaler = StandardScaler()
        scaled_columns  = scaler.fit_transform(edge_features_df[EDGE_FEATURES_TO_SCALE_COLUMNS]) 
        encoded_columns = edge_features_df[EDGE_FEATURES_ENCODED_COLUMNS]
        edge_features_scaled = np.concatenate([scaled_columns, encoded_columns], axis=1)
        # PCA
        pca = PCA(n_components=0.95)
        pca.fit(edge_features_scaled)
        edge_features_reduced = pca.transform(edge_features_scaled)

        return node_features_reduced, edge_features_reduced

# ------------------ UTILS ------------------

    def normalize_adjacency(self, adjacency_matrixes: t.List[sp.csr_matrix]) -> None:
        """Normalizes the adjacency matrixes in the input list:
        * Adds the identity
        * Normalizes each line by the inverse degree of its corresponding node

        :param adjacency_matrix: The list of adjacency matrixes to normalize
        :returns: None, the matrixes are normalized in place in the list.
        """
        for graph_index in range(len(adjacency_matrixes)):
            adj = adjacency_matrixes[graph_index]
            n = adj.shape[0]
            adj = adj + sp.identity(n)
            degs = adj.dot(np.ones(n))
            inv_degs = np.power(degs, -1)
            diags = sp.diags(inv_degs)
            adj_normalized = diags.dot(adj)
            adjacency_matrixes[graph_index] = adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.csr_matrix) -> torch.tensor:
    """
    Converts a Scipy sparse matrix to a sparse Torch tensor.

    :param sparse_mx: The input Scipy sparse matrix
    :returns: The converted Torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

structure_data = StructureData()
