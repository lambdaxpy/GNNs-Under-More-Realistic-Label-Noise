import itertools
from pathlib import Path

import networkx as nx
import torch
import torch_geometric
from ml_collections.config_dict import config_dict
from torch import optim
from torch.nn import functional as F
import numpy as np
from torch_geometric.utils import to_networkx

from framework.datasets.loader import load_data_split
from framework.env import PATH
from framework.log.logger import get_info_logger
from framework.models.utils.lpm_data import preprocess_features, preprocess_adj, normalize_S, sample_mask
from framework.models.utils.lpm_utils import masked_loss, acc
import higher
from framework.models.utils.lpm_model import GCN, ANet
import os
import scipy.sparse as sp

from framework.models.utils.mapper import DATASET, OPTIMIZER
from framework.resultbuilder.csvbuilder import parse_df_into_csv
from framework.resultbuilder.dfbuilder import build_df_from_dict
from framework.train.utils.earlystopping import EarlyStopping

INFO_LOGGER = get_info_logger(name=__name__)


def get_clean_mask(num_nodes, num_classes, y_label, idx_val, clean_label_num, device):
    data_list_clean = {}
    for j in range(num_classes):
        data_list_clean[j] = [idx_val[i] for i, label in enumerate(y_label[idx_val]) if label == j]
    list_clean = []
    num = int(clean_label_num / num_classes)
    for i, ind in data_list_clean.items():
        np.random.shuffle(ind)
        list_clean.append(ind[0:num])
    idx_clean = np.array(list_clean)
    idx_clean = idx_clean.flatten()
    clean_mask = sample_mask(idx_clean, num_nodes)

    clean_mask = torch.from_numpy(clean_mask).bool().to(device)

    return clean_mask


def lpm_train_and_test(adj, features, y, train_mask, val_mask, test_mask, clean_label_num, hidden, ANet_dim, lr, A_lr,
                       weight_decay, A_weight_decay, epochs, lpa_iters, patience, hp_optimizer, val=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INFO_LOGGER.info(f"Evaluating LPM on {device}")

    adj = sp.coo_matrix(adj)
    y = torch.from_numpy(y)
    y_label = y.argmax(dim=1).to(device)
    num_classes = y.shape[1]
    num_nodes = y.shape[0]
    features = preprocess_features(features)
    supports = preprocess_adj(adj)

    idx_val = np.array([i for i in range(val_mask.size(dim=0)) if val_mask[i]])

    clean_mask = get_clean_mask(num_nodes, num_classes, y_label, idx_val, clean_label_num, device)  # get clean mask

    train_num = torch.sum(train_mask).item()

    i = torch.from_numpy(features[0]).long()
    v = torch.from_numpy(features[1])
    feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float().to(device)

    i = torch.from_numpy(supports[0]).long()
    v = torch.from_numpy(supports[1])
    support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

    num_features_nonzero = feature._nnz()
    feat_dim = feature.shape[1]

    net = GCN(feat_dim, hidden, num_classes, num_features_nonzero).to(device)
    A_Net = ANet(2, ANet_dim, 1).to(device)
    optimizer = OPTIMIZER[hp_optimizer](net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_ANet = OPTIMIZER[hp_optimizer](A_Net.parameters(), A_lr, weight_decay=A_weight_decay)
    eps = np.finfo(float).eps
    # y_real = y.argmax(dim=1).to(device)

    label_onehot = torch.zeros(num_nodes, num_classes).to(device)
    # convert noisy labels to one-hot label
    ones = torch.eye(num_classes).to(device)
    y_label_onehot = torch.zeros(num_nodes, num_classes).to(device)
    y_label_onehot[train_mask] = ones.index_select(0, y_label[train_mask])

    early_stopping = EarlyStopping(patience, Path(os.path.join(PATH, "train/esmodels/lpm.pt")))

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        out, _ = net((feature, support))

        v = torch.mean((out[adj.row] - out[adj.col]) ** 2, 1)
        v = 1 / (v + eps)
        v = v.detach().cpu().numpy()
        S = sp.coo_matrix((v, (adj.row, adj.col)), shape=adj.shape)
        S = normalize_S(S)

        label_onehot[clean_mask] = ones.index_select(0, y_label[clean_mask])
        label_sp = sp.lil_matrix(label_onehot.cpu().numpy())
        clean_mask_np = clean_mask.cpu().numpy()
        for j in range(lpa_iters):
            if j == 0:
                Z = S.dot(label_sp)
            else:
                Z = S.dot(Z)
            Z[clean_mask_np] = label_sp[clean_mask_np]
        Z = torch.from_numpy(Z.toarray()).to(device)
        Z = F.softmax(Z, dim=1)
        Z_train = Z.argmax(dim=1)[train_mask]
        tmp_mask = Z_train == y_label[train_mask]
        clean_mask[:train_num] = tmp_mask
        train_sel_mask = torch.zeros(y.shape[0]).bool()
        train_sel_mask[:train_num] = tmp_mask
        train_sel_mask = train_sel_mask.to(device)
        loss_1 = masked_loss(out, y_label, train_sel_mask)
        loss_1.backward()
        optimizer.step()
        train_left_mask = torch.zeros(y.shape[0]).bool().to(device)
        train_left_mask[:train_num] = ~tmp_mask

        with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_opt):
            meta_out, _ = meta_net((feature, support))
            meta_cost_1 = F.cross_entropy(meta_out[train_left_mask], y_label[train_left_mask], reduction='none')
            meta_cost_1 = torch.reshape(meta_cost_1, (len(meta_cost_1), 1))
            meta_cost_2 = torch.sum(F.softmax(meta_out[train_left_mask], 1) * (
                    F.log_softmax(meta_out[train_left_mask], 1) - torch.log(Z[train_left_mask])), 1)
            meta_cost_2 = torch.reshape(meta_cost_2, (len(meta_cost_2), 1))
            meta_cost_a = torch.cat((meta_cost_1, meta_cost_2), 1).float()
            a_lambda = A_Net(meta_cost_a)
            y_mul = a_lambda * y_label_onehot[train_left_mask] + (1 - a_lambda) * Z[train_left_mask]
            meta_cost = torch.sum(y_mul * (torch.log(y_mul) - F.log_softmax(meta_out[train_left_mask], 1)), 1).mean()
            meta_net.zero_grad()
            meta_opt.step(meta_cost)
            meta_out, _ = meta_net((feature, support))
            meta_loss = masked_loss(meta_out, y_label, clean_mask)
            optimizer_ANet.zero_grad()
            meta_loss.backward()
            optimizer_ANet.step()

        out, _ = net((feature, support))
        with torch.no_grad():
            lambda_new = A_Net(meta_cost_a)

        y_mul = lambda_new * y_label_onehot[train_left_mask] + (1 - lambda_new) * Z[train_left_mask]

        y_mul = y_mul.detach()
        loss_2 = torch.sum(
            F.softmax(out[train_left_mask], 1) * (F.log_softmax(out[train_left_mask], 1) - torch.log(y_mul)), 1).mean()
        loss_2.backward()
        optimizer.step()
        val_acc = acc(out, y_label, val_mask)

        if early_stopping(val_acc, net):
            INFO_LOGGER.info(f"Early Stopping at Epoch: {epoch + 1}")
            break

    net.eval()

    if val:
        out, _ = net((feature, support))
        val_acc = acc(out, y_label, val_mask)
        del net
        del A_Net
        return val_acc

    out, _ = net((feature, support))
    test_acc = acc(out, y_label, test_mask)
    del net
    del A_Net
    return test_acc


def optimize_hp(dataset: str, dataset_obj: torch_geometric.data.Dataset, hp: config_dict.ConfigDict, noise_type: str,
                adj, features, num_classes, y)\
        -> dict:
    INFO_LOGGER.info("Starting HP Optimization of LPM")

    lr = hp.lr
    patience = hp.patience
    weight_decay = hp["weight-decay"]
    hidden_dim = hp["hidden_dim"]
    epochs = hp.epochs
    clean_label_num = hp["clean_label_num"]
    lpa_iters = hp["lpa-iters"]
    a_weight_decay = hp["A_weight-decay"]
    a_lr = hp["A_lr"]
    a_net_dims = hp["ANet_dim"]


    hp_dict = hp.to_dict()

    df_dict = {"split": [], "ANet_dim": [], "lpa-iters": [], "accuracy": []}

    best_hp_dict = hp_dict.copy()
    best_accuracy = -1.0

    cartesian_product = itertools.product(a_net_dims, lpa_iters)

    for (a_net_dim, lpa_iter) in cartesian_product:
        INFO_LOGGER.info(f"Training for: ({a_net_dim}, {lpa_iter})")
        acc_sum = 0.
        for i in range(10):  # number of splits
            INFO_LOGGER.info(f"HP Optimization of LPM with: {i + 1}. Split")
            data = load_data_split(dataset, None, None, i + 1)
            INFO_LOGGER.info(f"Is Undirected: {data.is_undirected()}")
            accuracy = lpm_train_and_test(adj, features, y, data.train_mask, data.val_mask, data.test_mask, clean_label_num,
                                          hidden_dim, a_net_dim, lr, a_lr, weight_decay, a_weight_decay, epochs, lpa_iter,
                                          patience, hp.optimizer, val=True)
            INFO_LOGGER.info(f"Accuracy: {accuracy}")
            acc_sum += accuracy
            df_dict["ANet_dim"].append(a_net_dim)
            df_dict["lpa-iters"].append(lpa_iter)
            df_dict["accuracy"].append(accuracy)
            df_dict["split"].append(i + 1)

        avg_acc = acc_sum / 10
        if avg_acc > best_accuracy:
            best_hp_dict["ANet_dim"] = a_net_dim
            best_hp_dict["lpa-iters"] = lpa_iter
            best_accuracy = avg_acc

        INFO_LOGGER.info(f"LPM Average Validation Accuracy: {avg_acc}")

    df = build_df_from_dict(df_dict)
    parse_df_into_csv(df, os.path.join(PATH, f"output/hpresults/lpm_{dataset}_{noise_type}.csv"))
    INFO_LOGGER.info(f"Best HP for LPM: {best_hp_dict}")

    return best_hp_dict


if __name__ == '__main__':
    dataset_obj = DATASET["cora"](root="../datasets", name="cora", split="random", num_train_per_class=20, num_val=500,
                                  num_test=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset_obj[0].to(device)
    graph = to_networkx(data, node_attrs=["x"])
    adj = nx.adjacency_matrix(graph)
    num_classes = data.y.max().item() + 1
    y = np.array([[1 if i == label else 0 for i in range(num_classes)] for label in data.y])
    features = sp.csr_matrix(data.x.numpy())
    accuracy = lpm_train_and_test(adj, features, y, data.train_mask, data.val_mask, data.test_mask, 28, 16,
                       64, 0.01, 0.0004, 0.0005, 0.0001, 1000, 50,
                       50, "adam")
    print("Test Accuracy:", accuracy)