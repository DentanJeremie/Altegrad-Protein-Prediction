from src.utils.logging import logger
logger.warn('The code in this folder is a draft, untested and not integrated into the pipeline.')

import scipy.sparse as sp
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

from src.utils.pathtools import project

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
    path = str(project.data) + '/'
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
    
    
    #Process node attributes
    columns_encoded = list(range(3,23))
    columns_to_scale  = [0,1,2]+list(range(25,86))
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore")
    encoded_columns = ohe.fit_transform(x[:,[23,24]]).toarray()
    scaled_columns  = scaler.fit_transform(x[:,columns_to_scale]) 
    unmodified_columns = x[:,columns_encoded]
    x_pca = np.concatenate([scaled_columns, encoded_columns,unmodified_columns], axis= 1)
    pca = PCA(n_components=0.95)
    x_pca = pca.fit_transform(x_pca)

    #Process edge attributes
    columns_encoded = list(range(1,5))
    columns_to_scale  = [0]
    scaler = StandardScaler()
    scaled_columns  = scaler.fit_transform(edge_attr[:,columns_to_scale]) 
    encoded_columns = edge_attr[:,columns_encoded]
    edge_attr_pca = np.concatenate([scaled_columns, encoded_columns], axis=1)
    pca = PCA(n_components=0.95)
    pca.fit(edge_attr_pca)
    edge_attr_pca = pca.transform(edge_attr_pca)

    adj = []
    features = []
    edge_features = []
    edge_index = []
    pos = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
        #edge_features.append(edge_attr[idx_m:idx_m+adj[i].nnz,:])
        edge_features.append(edge_attr_pca[idx_m:idx_m+adj[i].nnz,0])
        features.append(x_pca[idx_n:idx_n+graph_size[i],:])
        edge_index_raw = torch.from_numpy(np.array([edges[idx_m:idx_m+adj[i].nnz][:,0], edges[idx_m:idx_m+adj[i].nnz][:,1]]))
        edge_index.append(edge_index_raw - edge_index_raw.min())
        idx_n += graph_size[i]
        idx_m += adj[i].nnz
    
    return adj, features, edge_features, edge_index