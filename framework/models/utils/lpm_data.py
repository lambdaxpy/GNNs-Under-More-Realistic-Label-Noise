import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))  # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten()  # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0.  # zero inf data
    r_mat_inv = sp.diags(r_inv)  # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features)  # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features)  # [coordinates, data, shape], []


def normalize_S(S):
    """Symmetrically normalize similarity matrix."""
    S = sp.coo_matrix(S)
    rowsum = np.array(S.sum(1))  # D
    for i in range(len(rowsum)):
        if rowsum[i] == 0:
            print("error")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return S.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5SD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_S(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
