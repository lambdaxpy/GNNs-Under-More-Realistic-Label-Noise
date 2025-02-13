# %%
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.env import PATH
from framework.log.logger import get_info_logger, get_error_logger
from framework.models.gcn import Model
from framework.models.utils.mapper import DATASET, OPTIMIZER
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import framework.models.dgnnutils as dgnnutils

from framework.train.utils.earlystopping import EarlyStopping
from pathlib import Path


INFO_LOGGER = get_info_logger(name=__name__)
ERROR_LOGGER = get_error_logger(name=__name__)


class NoiseAda(nn.Module):
    def __init__(self, class_size, noise):
        super(NoiseAda, self).__init__()
        # Noise Transition Matrix
        P = torch.FloatTensor(dgnnutils.build_uniform_P(class_size, noise))
        self.B = torch.nn.parameter.Parameter(torch.log(P))

    def forward(self, pred):
        P = F.softmax(self.B, dim=1)
        return pred @ P


class S_model(Model):
    def __init__(self, nlayers, nfeat, nhid, nclass, hp_optimizer, lr=0.001, weight_decay=0.0001, norm="none",
                 dropout=0.5, res=False, patience=50, noise_t=0.2):
        super(S_model, self).__init__("gcn", nlayers, nfeat, nhid, nclass, 0, 0,
                                      normalization=norm, dropout=dropout, res=res)
        self.noise_ada = NoiseAda(nclass, noise_t)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hp_optimizer = hp_optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience

    def evaluate_dgnn_with_val(self, data, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        self.fit(features, adj, labels, idx_train, idx_val, train_iters, verbose)
        # From now on state_dict is set with trained parameters.

        self.eval()

        pred = self.forward(self.features, self.edge_index)
        pred = F.log_softmax(pred, dim=1).argmax(dim=1)
        correct = (pred[idx_val] == data.y[idx_val]).sum()
        accuracy = int(correct) / int(data.x[idx_val].size()[0])

        return accuracy

    def evaluate_dgnn_with_test(self, data, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=200,
                                verbose=False):
        self.fit(features, adj, labels, idx_train, idx_val, train_iters, verbose)
        # From now on state_dict is set with trained parameters.

        self.eval()

        pred = self.forward(self.features, self.edge_index)
        pred = F.log_softmax(pred, dim=1).argmax(dim=1)
        correct = (pred[idx_test] == data.y[idx_test]).sum()
        accuracy = int(correct) / int(data.x[idx_test].size()[0])

        return accuracy

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        # self.device = self.gc1.weight.device
        # self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = dgnnutils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features.cpu()))

        self.features = features.to(self.device)

        self.labels = torch.LongTensor(np.array(labels.cpu())).to(self.device)

        self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        self.to(self.device)
        INFO_LOGGER.info(f"Evaluating D-GNN on {self.device}")
        optimizer = OPTIMIZER[self.hp_optimizer](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = EarlyStopping(self.patience, Path(os.path.join(PATH, "train/esmodels/dgnn.pt")))
        stopped_early = False
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index)
            pred = F.softmax(output, dim=1)
            eps = 1e-8
            score = self.noise_ada(pred).clamp(eps, 1 - eps)

            loss_train = F.cross_entropy(torch.log(score[idx_train]), self.labels[idx_train])
            loss_train.backward()

            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.edge_index)
            acc_val = dgnnutils.accuracy(output[idx_val], labels[idx_val])

            if verbose:
                INFO_LOGGER.info('Epoch {}, val acc: {}'.format(i, acc_val))

            if early_stopping(acc_val, self):
                INFO_LOGGER.info(f"Early Stopping at Epoch: {i}")
                stopped_early = True
                break

        if stopped_early:
            self.load_state_dict(torch.load(os.path.join(PATH, "train/esmodels/dgnn.pt")))

        # Call Destructor of Early Stopping.
        del early_stopping
# %%
