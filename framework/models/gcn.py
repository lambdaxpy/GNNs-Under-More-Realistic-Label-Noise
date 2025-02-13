import os
import itertools

import torch
import torch.nn.functional as F
import torch_geometric.data
from torch import nn
from torch_geometric.data.dataset import Dataset
from torch_geometric.transforms import to_undirected
from ml_collections import config_dict

from framework.models.sam import SAM
from framework.train.utils.earlystopping import EarlyStopping
from framework.models.utils.modules import FeedForwardModule, GCNModule, ResidualModuleWrapper
from framework.models.utils.mapper import OPTIMIZER, DATASET
from framework.log.logger import get_info_logger, get_error_logger
from framework.configparser.yamlparser import parse_yaml_file_to_config_dict
from framework.resultbuilder.dfbuilder import build_df_from_dict
from framework.resultbuilder.csvbuilder import parse_df_into_csv
from framework.datasets.loader import load_data_split
from framework.env import PATH

from pathlib import Path

INFO_LOGGER = get_info_logger(name=__name__)
ERROR_LOGGER = get_error_logger(name=__name__)

MODULES = {
    'mlp': [FeedForwardModule],
    'gcn': [GCNModule],
}

NORMALIZATION = {
    'none': nn.Identity,
    'layer': nn.LayerNorm,
    'batch': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout, res):
        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                if res:
                    residual_module = ResidualModuleWrapper(module=module,
                                                            normalization=normalization,
                                                            dim=hidden_dim,
                                                            hidden_dim_multiplier=hidden_dim_multiplier,
                                                            num_heads=num_heads,
                                                            dropout=dropout)
                else:
                    residual_module = module(dim=hidden_dim,
                                             normalization=normalization,
                                             hidden_dim_multiplier=hidden_dim_multiplier,
                                             num_heads=num_heads,
                                             dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x, edge_index):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(x, edge_index)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)
        return x



def evaluate_gcn_with_val(dataset: Dataset, data: torch_geometric.data.Data, hp_optimizer: str, lr: float = 0.001,
                          weight_decay: float = 0.01, hidden_dim: int = 256, dropout: float = 0.0, norm: str = "none",
                          layer: int = 2, res: bool = False, epochs: int = 1000, patience: int = 50, sam: bool = False,
                          rho: float = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn = Model("gcn", layer, data.num_node_features, hidden_dim, dataset.num_classes,
                0, 0, norm, dropout, res)
    gcn.to(device)
    if not sam:
        INFO_LOGGER.info(f"Evaluating GCN on {device}")
    else:
        INFO_LOGGER.info(f"Evaluating GCN/SAM on {device}")
    optimizer = OPTIMIZER[hp_optimizer](gcn.parameters(), lr=lr, weight_decay=weight_decay)

    if sam:
        optimizer = OPTIMIZER[hp_optimizer]
        optimizer = SAM(gcn.parameters(), optimizer, lr=lr, weight_decay=weight_decay, rho=rho)

    early_stopping = EarlyStopping(patience, Path(os.path.join(PATH, "train/esmodels/gcn.pt")))
    gcn.train()
    stopped_early = False
    for i in range(epochs):
        if not sam:
            optimizer.zero_grad()
            out = gcn(data.x, data.edge_index)
            out_softmax = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out_softmax[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        else:
            # First Forward Pass for SAM.
            out = gcn(data.x, data.edge_index)
            out_softmax = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out_softmax[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # Second Forward Pass for SAM.
            out = gcn(data.x, data.edge_index)
            out_softmax = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out_softmax[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.second_step(zero_grad=True)

        # Check Validation Accuracy For Early Stopping
        val_pred = out_softmax.argmax(dim=1)
        correct = (val_pred[data.val_mask] == data.y[data.val_mask]).sum()
        val_accuracy = int(correct) / int(data.x[data.val_mask].size()[0])
        if early_stopping(val_accuracy, gcn):
            INFO_LOGGER.info(f"Early Stopping at Epoch: {i}")
            stopped_early = True
            break

    if stopped_early:
        gcn.load_state_dict(torch.load(os.path.join(PATH, "train/esmodels/gcn.pt")))

    # Call Destructor of Early Stopping.
    del early_stopping

    gcn.eval()
    pred = gcn(data.x, data.edge_index)
    pred = F.log_softmax(pred, dim=1).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    accuracy = int(correct) / int(data.x[data.val_mask].size()[0])

    return accuracy


def evaluate_gcn_with_test(dataset: Dataset, data: torch_geometric.data.Data, hp_optimizer: str, lr: float = 0.001,
                           weight_decay: float = 0.01, hidden_dim: int = 256, dropout: float = 0.0, norm: str = "none",
                           layer: int = 2, res: bool = False, epochs: int = 1000, patience: int = 50, sam: bool = False,
                           rho: float = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn = Model("gcn", layer, data.num_node_features, hidden_dim, dataset.num_classes,
                0, 0, norm, dropout, res)
    gcn.to(device)
    INFO_LOGGER.info(f"Evaluating GCN on {device}")
    optimizer = OPTIMIZER[hp_optimizer](gcn.parameters(), lr=lr, weight_decay=weight_decay)

    if sam:
        optimizer = OPTIMIZER[hp_optimizer]
        optimizer = SAM(gcn.parameters(), optimizer, lr=lr, weight_decay=weight_decay, rho=rho)

    early_stopping = EarlyStopping(patience, Path(os.path.join(PATH, "train/esmodels/gcn.pt")))
    gcn.train()
    stopped_early = False
    for i in range(epochs):
        if not sam:
            optimizer.zero_grad()
            out = gcn(data.x, data.edge_index)
            out_softmax = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out_softmax[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        else:
            # First Forward Pass for SAM.
            out = gcn(data.x, data.edge_index)
            out_softmax = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out_softmax[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # Second Forward Pass for SAM.
            out = gcn(data.x, data.edge_index)
            out_softmax = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out_softmax[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.second_step(zero_grad=True)

        # Check Validation Accuracy For Early Stopping
        val_pred = out_softmax.argmax(dim=1)
        correct = (val_pred[data.val_mask] == data.y[data.val_mask]).sum()
        val_accuracy = int(correct) / int(data.x[data.val_mask].size()[0])
        if early_stopping(val_accuracy, gcn):
            INFO_LOGGER.info(f"Early Stopping at Epoch: {i}")
            stopped_early = True
            break

    if stopped_early:
        gcn.load_state_dict(torch.load(os.path.join(PATH, "train/esmodels/gcn.pt")))

    # Call Destructor of Early Stopping.
    del early_stopping

    gcn.eval()
    pred = gcn(data.x, data.edge_index)
    pred = F.log_softmax(pred, dim=1).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct) / int(data.x[data.test_mask].size()[0])

    return accuracy


def optimize_hp(dataset: str, dataset_obj: torch_geometric.data.Dataset, hp: config_dict.ConfigDict, noise_type: str) \
        -> dict:
    INFO_LOGGER.info("Starting HP Optimization of GCN")

    hp_optimizer = hp.optimizer
    lr = hp.lr
    patience = hp.patience
    weight_decay = hp["weight-decay"]
    hidden_dims = hp["hidden_dim"]
    dropout_values = hp.dropout
    norms = hp.norm
    layers = hp.layers
    res_bools = hp.res
    epochs = hp.epochs

    hp_dict = hp.to_dict()

    df_dict = {"split": [], "hidden_dim": [], "dropout": [],
               "norm": [], "layers": [], "res": [], "accuracy": []}

    best_hp_dict = hp_dict.copy()
    best_accuracy = -1.0

    cartesian_product = itertools.product(hidden_dims, dropout_values, norms, layers, res_bools)

    for (hidden_dim, dropout, norm, layer, res) in cartesian_product:
        INFO_LOGGER.info(f"Training for: ({hidden_dim}, {dropout}, {norm}, {layer}, {res})")
        acc_sum = 0.
        for i in range(10):  # number of splits
            INFO_LOGGER.info(f"HP Optimization of GCN with: {i + 1}. Split")
            data = load_data_split(dataset, None, None, i + 1)
            INFO_LOGGER.info(f"Is Undirected: {data.is_undirected()}")
            accuracy = evaluate_gcn_with_val(dataset_obj, data, hp_optimizer, lr, weight_decay, hidden_dim, dropout,
                                             norm, layer, res, epochs, patience, False)
            INFO_LOGGER.info(f"Accuracy: {accuracy}")
            acc_sum += accuracy
            df_dict["hidden_dim"].append(hidden_dim)
            df_dict["dropout"].append(dropout)
            df_dict["norm"].append(norm)
            df_dict["layers"].append(layer)
            df_dict["res"].append(res)
            df_dict["accuracy"].append(accuracy)
            df_dict["split"].append(i + 1)

        avg_acc = acc_sum / 10
        if avg_acc > best_accuracy:
            best_hp_dict["hidden_dim"] = hidden_dim
            best_hp_dict["dropout"] = dropout
            best_hp_dict["norm"] = norm
            best_hp_dict["layers"] = layer
            best_hp_dict["res"] = res
            best_accuracy = avg_acc

        INFO_LOGGER.info(f"GCN Average Validation Accuracy: {avg_acc}")

    df = build_df_from_dict(df_dict)
    parse_df_into_csv(df, os.path.join(PATH, f"output/hpresults/gcn_{dataset}_{noise_type}.csv"))
    INFO_LOGGER.info(f"Best HP for GCN: {best_hp_dict}")

    return best_hp_dict


def optimize_hp_with_sam(dataset: str, dataset_obj: torch_geometric.data.Dataset, hp: config_dict.ConfigDict,
                         noise_type: str) -> dict:
    INFO_LOGGER.info("Starting HP Optimization of GCN/SAM")
    hp_optimizer = hp.optimizer
    lr = hp.lr
    patience = hp.patience
    weight_decay = hp["weight-decay"]
    hidden_dim = hp["hidden_dim"]
    dropout = hp.dropout
    norm = hp.norm
    layer = hp.layers
    res = hp.res
    rhos = hp.rho
    epochs = hp.epochs

    hp_dict = hp.to_dict()

    df_dict = {"split": [], "rho": [], "accuracy": []}

    best_hp_dict = hp_dict.copy()
    best_accuracy = -1.0

    for rho in rhos:
        INFO_LOGGER.info(f"Training for: {rho}")
        acc_sum = 0.
        for i in range(10):  # number of splits
            INFO_LOGGER.info(f"HP Optimization of GCN/SAM with: {i + 1}. Split")
            data = load_data_split(dataset, None, None, i + 1)
            INFO_LOGGER.info(f"Is Undirected: {data.is_undirected()}")
            accuracy = evaluate_gcn_with_val(dataset_obj, data, hp_optimizer, lr, weight_decay, hidden_dim, dropout,
                                             norm, layer, res, epochs, patience, True, rho)
            INFO_LOGGER.info(f"Accuracy: {accuracy}")
            acc_sum += accuracy
            df_dict["rho"].append(rho)
            df_dict["accuracy"].append(accuracy)
            df_dict["split"].append(i + 1)

        avg_acc = acc_sum / 10
        if avg_acc > best_accuracy:
            best_hp_dict["rho"] = rho
            best_accuracy = avg_acc

        INFO_LOGGER.info(f"GCN/SAM Average Validation Accuracy: {avg_acc}")

    df = build_df_from_dict(df_dict)
    parse_df_into_csv(df, os.path.join(PATH, f"output/hpresults/gcn_{dataset}_{noise_type}.csv"))
    INFO_LOGGER.info(f"Best HP for GCN: {best_hp_dict}")

    return best_hp_dict


if __name__ == '__main__':
    config = parse_yaml_file_to_config_dict(os.path.join(PATH, "tests/testconfig.yaml"))
    dataset_obj = DATASET["cora"](root="../datasets", name="cora", split="random", num_train_per_class=20, num_val=500,
                                  num_test=1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset_obj[0].to(device)
    print(optimize_hp("cora", dataset_obj, config.model, "uniform"))
