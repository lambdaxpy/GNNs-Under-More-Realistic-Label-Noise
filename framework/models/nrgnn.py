# %%
import itertools
import time

import networkx as nx
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import os

import torch_geometric
import torch_geometric.utils as utils
import scipy.sparse as sp
from ml_collections.config_dict import config_dict
from torch_geometric.utils import to_networkx

from framework.datasets.loader import load_data_split
from framework.env import PATH
from framework.log.logger import get_info_logger
from framework.models.utils.nrgnn_gcn import GCN

from framework.models.utils.mapper import DATASET, OPTIMIZER
from framework.models.dgnnutils import accuracy, sparse_mx_to_torch_sparse_tensor
from framework.resultbuilder.csvbuilder import parse_df_into_csv
from framework.resultbuilder.dfbuilder import build_df_from_dict

INFO_LOGGER = get_info_logger(name=__name__)

class NRGNN:
    def __init__(self, hidden_dim, dropout, alpha, beta, n_p, p_u, n_n, t_small, edge_hidden, debug):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.n_p = n_p
        self.p_u = p_u
        self.n_n = n_n
        self.t_small = t_small
        self.edge_hidden = edge_hidden
        self.debug = debug
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None

        self.early_stopping = False
        self.model_counter = 0

    def evaluate_nrgnn_with_test(self, features, adj, labels, idx_train, idx_val, idx_test, epochs, lr, weight_decay,
                                 hp_optimizer, patience):
        INFO_LOGGER.info(f"Evaluating NRGNN on {self.device}")
        self.fit(features, adj, labels, idx_train, idx_val, epochs, lr, weight_decay, hp_optimizer, patience)
        return self.test(idx_test)

    def evaluate_nrgnn_with_val(self, features, adj, labels, idx_train, idx_val, idx_test, epochs, lr, weight_decay,
                                 hp_optimizer, patience):
        INFO_LOGGER.info(f"Evaluating NRGNN on {self.device}")
        self.fit(features, adj, labels, idx_train, idx_val, epochs, lr, weight_decay, hp_optimizer, patience)
        return self.test(idx_val)

    def fit(self, features, adj, labels, idx_train, idx_val, epochs, lr, weight_decay,
            hp_optimizer, patience):
        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features.cpu()))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels.cpu())).to(self.device)

        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)

        self.predictor = GCN(nfeat=features.shape[1],
                             nhid=self.hidden_dim,
                             nclass=labels.max().item() + 1,
                             self_loop=True,
                             dropout=self.dropout, device=self.device).to(self.device)

        self.model = GCN(nfeat=features.shape[1],
                         nhid=self.hidden_dim,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.dropout, device=self.device).to(self.device)

        self.estimator = EstimateAdj(features.shape[1], idx_train, self.edge_hidden, self.t_small, self.n_n,
                                     device=self.device).to(self.device)

        # obtain the condidate edges linking unlabeled and labeled nodes
        self.pred_edge_index = self.get_train_edge(edge_index, features, self.n_p, idx_train)

        self.optimizer = OPTIMIZER[hp_optimizer](
            list(self.model.parameters()) + list(self.estimator.parameters()) + list(self.predictor.parameters()),
            lr=lr, weight_decay=weight_decay)

        # Train model
        # t_total = time.time()
        for epoch in range(epochs):
            self.train(epoch, features, edge_index, idx_train, idx_val, patience)
            if self.early_stopping:
                INFO_LOGGER.info(f"Early Stopping at Epoch: {epoch + 1}")
                break

        # print("Optimization Finished!")
        # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        # print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        self.predictor.load_state_dict(self.predictor_model_weigths)

    def train(self, epoch, features, edge_index, idx_train, idx_val, patience):
        # t = time.time()
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        # obtain representations and rec loss of the estimator
        representations, rec_loss = self.estimator(edge_index, features)

        # prediction of accurate pseudo label miner
        predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index, representations)
        pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
        predictor_weights = torch.cat([torch.ones([edge_index.shape[1]], device=self.device), predictor_weights], dim=0)

        log_pred = self.predictor(features, pred_edge_index, predictor_weights)

        # obtain accurate pseudo labels and new candidate edges
        if self.best_pred == None:
            pred = F.softmax(log_pred, dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
        else:
            pred = self.best_pred

        # prediction of the GCN classifier
        estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index, representations)
        estimated_weights = torch.cat([predictor_weights, estimated_weights], dim=0)
        model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)
        output = self.model(features, model_edge_index, estimated_weights)
        pred_model = F.softmax(output, dim=1)

        eps = 1e-8
        pred_model = pred_model.clamp(eps, 1 - eps)

        # loss from pseudo labels
        loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()

        # loss of accurate pseudo label miner
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])

        # loss of GCN classifier
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])

        total_loss = loss_gcn + loss_pred + self.alpha * rec_loss + self.beta * loss_add
        total_loss.backward()

        self.optimizer.step()

        acc_train = accuracy(output[idx_train].detach(), self.labels[idx_train])

        # Evaluate validation set performance separately,
        self.model.eval()
        self.predictor.eval()
        pred = F.softmax(self.predictor(features, pred_edge_index, predictor_weights), dim=1)
        output = self.model(features, model_edge_index, estimated_weights.detach())

        acc_pred_val = accuracy(pred[idx_val], self.labels[idx_val])
        acc_val = accuracy(output[idx_val], self.labels[idx_val])

        if acc_pred_val > self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = estimated_weights.detach()
            self.best_model_index = model_edge_index
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                INFO_LOGGER.info('\t=== saving current graph/gcn, best_val_acc: {:.4f}'.format(self.best_val_acc.item()))
            self.model_counter = 0
        else:
            self.model_counter += 1
            if self.model_counter >= patience:
                self.early_stopping = True

        if self.debug:
            if epoch % 1 == 0:
                INFO_LOGGER.info('Epoch: {:04d}'.format(epoch + 1) +
                                 ' loss_gcn: {:.4f}'.format(loss_gcn.item()) +
                                 ' loss_pred: {:.4f}'.format(loss_pred.item()) +
                                 ' loss_add: {:.4f}'.format(loss_add.item()) +
                                 ' rec_loss: {:.4f}'.format(rec_loss.item()) +
                                 ' loss_total: {:.4f}'.format(total_loss.item()))
                INFO_LOGGER.info('Epoch: {:04d}'.format(epoch + 1) +
                                 ' acc_train: {:.4f}'.format(acc_train.item()) +
                                 ' acc_val: {:.4f}'.format(acc_val.item()) +
                                 ' acc_pred_val: {:.4f}'.format(acc_pred_val.item()))
                INFO_LOGGER.info('Size of add idx is {}'.format(len(self.idx_add)))

    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        features = self.features
        labels = self.labels

        self.predictor.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = torch.cat([self.edge_index, self.pred_edge_index], dim=1)
        output = self.predictor(features, pred_edge_index, estimated_weights)
        # loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        # acc_test = accuracy(output[idx_test], labels[idx_test])

        self.model.eval()
        estimated_weights = self.best_graph
        model_edge_index = self.best_model_index
        output = self.model(features, model_edge_index, estimated_weights)
        # loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        return float(acc_test)

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        '''
        obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
        n_p is the top n_p labeled nodes similar with unlabeled nodes
        '''

        if n_p == 0:
            return None

        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1, edge_index[0] == i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i], features[idx_train].T),
                                features[i].norm() * features[idx_train].norm(dim=1))
                _, rank = sim.topk(n_p)
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices)
                else:
                    indices = set()
                indices = indices - set(edge_index[1, edge_index[0] == i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges, len(features)).to(self.device)

        return poten_edges

    def get_model_edge(self, pred):

        idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.p_u)]

        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel), 1).T.flatten()
        mask = (row != col)
        unlabel_edge_index = torch.stack([row[mask], col[mask]], dim=0)

        return unlabel_edge_index, idx_add


# %%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, nfea, idx_train, edge_hidden, t_small, n_n, device='cuda'):
        super(EstimateAdj, self).__init__()

        self.edge_hidden = edge_hidden
        self.t_small = t_small
        self.n_n = n_n
        self.estimator = GCN(nfea, self.edge_hidden, self.edge_hidden, dropout=0.0, device=device)
        self.device = device
        self.representations = 0

    def forward(self, edge_index, features):
        representations = self.estimator(features, edge_index, \
                                         torch.ones([edge_index.shape[1]]).to(self.device).float())
        rec_loss = self.reconstruct_loss(edge_index, representations)

        return representations, rec_loss

    def get_estimated_weigths(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)

        estimated_weights = F.relu(output.detach())
        estimated_weights[estimated_weights < self.t_small] = 0.0

        return estimated_weights

    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=self.n_n * num_nodes)
        randn = randn[:, randn[0] < randn[1]]

        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0, neg1), dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0, pos1), dim=1)

        rec_loss = (F.mse_loss(neg, torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                   * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss


def optimize_hp(dataset: str, dataset_obj: torch_geometric.data.Dataset, hp: config_dict.ConfigDict, noise_type: str,
                graph, adj)\
        -> dict:
    INFO_LOGGER.info("Starting HP Optimization of NRGNN")

    hp_optimizer = hp.optimizer
    lr = hp.lr
    patience = hp.patience
    dropout = hp.dropout
    weight_decay = hp["weight-decay"]
    hidden_dim = hp["hidden_dim"]
    epochs = hp.epochs
    edge_hidden = hp["edge_hidden"]
    t_small = hp["t_small"]
    p_u = hp["p_u"]
    n = hp["n"]
    alphas = hp["alpha"]
    betas = hp["beta"]

    hp_dict = hp.to_dict()

    df_dict = {"split": [], "alpha": [], "beta": [], "accuracy": []}

    best_hp_dict = hp_dict.copy()
    best_accuracy = -1.0

    cartesian_product = itertools.product(alphas, betas)

    for (alpha, beta) in cartesian_product:
        INFO_LOGGER.info(f"Training for: ({alpha}, {beta})")
        acc_sum = 0.
        for i in range(10):  # number of splits
            INFO_LOGGER.info(f"HP Optimization of NRGNN with: {i + 1}. Split")
            data = load_data_split(dataset, None, None, i + 1)
            INFO_LOGGER.info(f"Is Undirected: {data.is_undirected()}")
            nrgnn = NRGNN(hidden_dim, dropout, alpha, beta, n, p_u, n, t_small, edge_hidden, False)
            idx_train = np.array([i for i in range(data.train_mask.size(dim=0)) if
                                  data.train_mask[i]])
            idx_val = np.array([i for i in range(data.val_mask.size(dim=0)) if data.val_mask[i]])
            idx_test = np.array([i for i in range(data.test_mask.size(dim=0)) if data.test_mask[i]])
            accuracy = nrgnn.evaluate_nrgnn_with_val(data.x, adj, data.y, idx_train,
                                                     idx_val, idx_test, epochs, lr, weight_decay,
                                                     hp.optimizer, patience)
            INFO_LOGGER.info(f"Accuracy: {accuracy}")
            acc_sum += accuracy
            df_dict["alpha"].append(alpha)
            df_dict["beta"].append(beta)
            df_dict["accuracy"].append(accuracy)
            df_dict["split"].append(i + 1)

        avg_acc = acc_sum / 10
        if avg_acc > best_accuracy:
            best_hp_dict["alpha"] = alpha
            best_hp_dict["beta"] = beta
            best_accuracy = avg_acc

        INFO_LOGGER.info(f"NRGNN Average Validation Accuracy: {avg_acc}")

    df = build_df_from_dict(df_dict)
    parse_df_into_csv(df, os.path.join(PATH, f"output/hpresults/nrgnn_{dataset}_{noise_type}.csv"))
    INFO_LOGGER.info(f"Best HP for NRGNN: {best_hp_dict}")

    return best_hp_dict


# %%

if __name__ == "__main__":
    start_time = time.time()
    dataset_obj = DATASET["cora"](root="../datasets", name="cora", split="random", num_train_per_class=20, num_val=500,
                                  num_test=1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset_obj[0].to(device)
    torch.autograd.set_detect_anomaly(True)
    nrgnn = NRGNN(16, 0.0, 0.03, 1, 50, 0.8, 50,
                  0.1, 64, True)
    graph = to_networkx(data, node_attrs=["x"])
    adj = nx.adjacency_matrix(graph)
    idx_train = np.array([i for i in range(data.train_mask.size(dim=0)) if data.train_mask[i]])
    idx_val = np.array([i for i in range(data.val_mask.size(dim=0)) if data.val_mask[i]])
    idx_test = np.array([i for i in range(data.test_mask.size(dim=0)) if data.test_mask[i]])
    accuracy = nrgnn.evaluate_nrgnn_with_test(data.x, adj, data.y, idx_train, idx_val, idx_test, 1000, 0.001,
                                   0.0, "adam", 50)

    print("Time elapsed: ", time.time() - start_time)

    print("Test Accuracy:", accuracy)

