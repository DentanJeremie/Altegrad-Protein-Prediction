import scipy.sparse as sp
import numpy as np
import torch


def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A = A + sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized

def load_data(): 
    """
    Function that loads graphs
    """  
    path = '/Users/meryem/Documents/GitHub/Altegrad-Protein-Prediction/data/'
    graph_indicator = np.loadtxt(path +"graph_indicator.txt", dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(path+"edgelist.txt", dtype=np.int64, delimiter=",")
    edges_inv = np.vstack((edges[:,1], edges[:,0]))
    edges = np.vstack((edges, edges_inv.T))
    s = edges[:,0]*graph_indicator.size + edges[:,1]
    idx_sort = np.argsort(s)
    edges = edges[idx_sort,:]
    edges,idx_unique =  np.unique(edges, axis=0, return_index=True)
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))

    x = np.loadtxt(path+"node_attributes.txt", delimiter=",")
    edge_attr = np.loadtxt(path+"edge_attributes.txt", delimiter=",")
    edge_attr = np.vstack((edge_attr,edge_attr))
    edge_attr = edge_attr[idx_sort,:]
    edge_attr = edge_attr[idx_unique,:]

    adj = []
    features = []
    edge_features = []
    edge_index = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
        edge_features.append(edge_attr[idx_m:idx_m+adj[i].nnz,:])
        features.append(x[idx_n:idx_n+graph_size[i],:])
        edge_index.append(torch.from_numpy(np.array([edges[idx_m:idx_m+adj[i].nnz][:,0], edges[idx_m:idx_m+adj[i].nnz][:,1]])))
        idx_n += graph_size[i]
        idx_m += adj[i].nnz
    
    return adj, features, edge_features, edge_index