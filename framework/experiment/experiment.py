import os.path
import shutil
import time

import networkx as nx
import numpy as np
import torch.cuda
from torch_geometric.utils import to_networkx

import framework.models.gcn as gcn
import framework.models.nrgnn as nrgnn_file
import framework.models.lpm as lpm_file

from ml_collections import config_dict
from framework.configparser.yamlparser import parse_hp_dict_to_yaml
from framework.models import pown
from framework.models.dgnn import S_model
from framework.models.lpm import lpm_train_and_test
from framework.models.nrgnn import NRGNN

from framework.models.utils.mapper import DATASET
from framework.datasets.loader import load_data_split
from framework.log.logger import get_info_logger, get_error_logger
from framework.env import PATH

import gc
import random
import scipy.sparse as sp


INFO_LOGGER = get_info_logger(name=__name__)
ERROR_LOGGER = get_error_logger(name=__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Experiment:
    def __init__(self, dataset: str, model: str, noise: str, noise_ratios: list[float], hp: config_dict.ConfigDict,
                 config: config_dict.ConfigDict):
        self.dataset = dataset
        self.model = model
        self.noise = noise
        self.noise_ratios = noise_ratios
        self.hp = hp
        self.config = config
        self.dataset_obj = None

        if self.dataset in ["cora", "citeseer", "pubmed"]:
            self.dataset_obj = DATASET[dataset](root=os.path.join(PATH, "datasets"), name=dataset, split="random",
                                                num_train_per_class=20,
                                                num_val=500, num_test=1000)
        else:
            self.dataset_obj = DATASET[dataset](root=os.path.join(PATH, "datasets"), name=dataset)

    def __repr__(self):
        return self.config.__repr__()

    def evaluate_experiment(self) -> [float]:  # accuracy for each noise ratio
        accuracy_list = []
        std_list = []
        for noise_ratio in self.noise_ratios:  # iterate over all noise ratios
            if self.noise != "uniform" and noise_ratio == 0.0:  # Only evaluate 0.0 noise ratio once uniform setting.
                continue
            INFO_LOGGER.info(f"Training with Noise Ratio: {noise_ratio}")
            acc_noise = 0.
            std_noise = 0.
            INFO_LOGGER.info(f"Using {self.model} with Best HP: {self.hp}")
            for i in range(self.hp.runs):  # amount of runs per noise ratio
                set_seed(42 + i)
                acc_run = 0.
                acc_run_list = []
                for j in range(10):  # 10 splits per noise ratio
                    accuracy = None
                    noisy_data = None
                    if self.noise is not None and noise_ratio > 0.:
                        if self.noise == "uniform":
                            INFO_LOGGER.info("Using Uniform Noise")
                            noisy_data = load_data_split(self.dataset, self.noise, noise_ratio, j + 1)
                        elif self.noise == "feature-based pair":
                            INFO_LOGGER.info("Using Feature-based Pair Noise")
                            noisy_data = load_data_split(self.dataset, self.noise, noise_ratio, j + 1)
                        elif self.noise == "structure-based pair":
                            INFO_LOGGER.info("Using Structure-based Pair Noise")
                            noisy_data = load_data_split(self.dataset, self.noise, noise_ratio, j + 1)
                    else:
                        noisy_data = load_data_split(self.dataset, None, None, j + 1)
                    lr = self.hp["lr"]
                    weight_decay = self.hp["weight-decay"]
                    hidden_dim = self.hp["hidden_dim"]
                    epochs = self.hp["epochs"]
                    patience = self.hp["patience"]
                    if self.model == "gcn" or self.model == "sam" or self.model == "dgnn" or self.model == "pown":
                        dropout = self.hp["dropout"]
                        norm = self.hp["norm"]
                        layer = self.hp["layers"]
                        res = self.hp["res"]
                        if self.model == "sam":
                            rho = self.hp["rho"]
                            accuracy = gcn.evaluate_gcn_with_test(self.dataset_obj, noisy_data, self.hp.optimizer, lr,
                                                                  weight_decay, hidden_dim, dropout, norm, layer, res,
                                                                  epochs, patience, True, rho)
                        elif self.model == "gcn":
                            accuracy = gcn.evaluate_gcn_with_test(self.dataset_obj, noisy_data, self.hp.optimizer, lr,
                                                                  weight_decay, hidden_dim, dropout, norm, layer, res,
                                                                  epochs, patience, False, None)
                        elif self.model == "pown":
                            sup_loss_weight = self.hp["sup_loss_weight"]
                            alpha = self.hp["alpha"]
                            u = self.hp["u"]
                            accuracy = pown.evaluate_pown(self.dataset_obj, noisy_data, noisy_data.test_mask,
                                                          self.hp.optimizer, lr,
                                                          weight_decay, hidden_dim, dropout, norm, layer, res,
                                                          epochs, patience, alpha, u, sup_loss_weight)

                        elif self.model == "dgnn":
                            noise_t = self.hp["noise_t"]
                            dgnn = S_model(layer, noisy_data.num_node_features, hidden_dim, self.dataset_obj.num_classes,
                                           self.hp.optimizer, lr, weight_decay, norm, dropout, res, patience, noise_t)
                            graph = to_networkx(noisy_data, node_attrs=["x"])
                            adj = nx.adjacency_matrix(graph)
                            idx_train = [i for i in range(noisy_data.train_mask.size(dim=0)) if noisy_data.train_mask[i]]
                            idx_val = [i for i in range(noisy_data.val_mask.size(dim=0)) if noisy_data.val_mask[i]]
                            idx_test = [i for i in range(noisy_data.test_mask.size(dim=0)) if noisy_data.test_mask[i]]
                            accuracy = dgnn.evaluate_dgnn_with_test(noisy_data, noisy_data.x, adj, noisy_data.y, idx_train,
                                                         idx_val=idx_val, idx_test=idx_test, train_iters=epochs,
                                                         verbose=False)
                            del dgnn  # Calling Destructor to clear GPU Memory.
                    elif self.model == "nrgnn":
                        dropout = self.hp["dropout"]
                        edge_hidden = self.hp["edge_hidden"]
                        t_small = self.hp["t_small"]
                        p_u = self.hp["p_u"]
                        n = self.hp["n"]
                        alpha = self.hp["alpha"]
                        beta = self.hp["beta"]
                        nrgnn = NRGNN(hidden_dim, dropout, alpha, beta, n,
                                      p_u, n, t_small, edge_hidden, False)
                        graph = to_networkx(noisy_data, node_attrs=["x"])
                        adj = nx.adjacency_matrix(graph)
                        idx_train = np.array([i for i in range(noisy_data.train_mask.size(dim=0)) if
                                     noisy_data.train_mask[i]])
                        idx_val = np.array([i for i in range(noisy_data.val_mask.size(dim=0)) if noisy_data.val_mask[i]])
                        idx_test = np.array([i for i in range(noisy_data.test_mask.size(dim=0)) if noisy_data.test_mask[i]])
                        accuracy = nrgnn.evaluate_nrgnn_with_test(noisy_data.x, adj, noisy_data.y, idx_train,
                                                                  idx_val, idx_test, epochs, lr, weight_decay,
                                                                  self.hp.optimizer, patience)
                        del nrgnn  # Calling Destructor to clear GPU Memory.
                    elif self.model == "lpm":
                        clean_label_num = self.hp["clean_label_num"]
                        lpa_iters = self.hp["lpa-iters"]
                        a_weight_decay = self.hp["A_weight-decay"]
                        a_lr = self.hp["A_lr"]
                        a_net_dim = self.hp["ANet_dim"]
                        graph = to_networkx(noisy_data, node_attrs=["x"])
                        adj = nx.adjacency_matrix(graph)
                        num_classes = noisy_data.y.max().item() + 1
                        y = np.array([[1 if i == label else 0 for i in range(num_classes)] for label in noisy_data.y])
                        features = sp.csr_matrix(noisy_data.x.cpu().numpy())
                        accuracy = lpm_train_and_test(adj, features, y, noisy_data.train_mask, noisy_data.val_mask,
                                                      noisy_data.test_mask, clean_label_num, hidden_dim,
                                                      a_net_dim, lr, a_lr, weight_decay,
                                                      a_weight_decay, epochs, lpa_iters, patience, self.hp.optimizer)

                    del noisy_data  # Calling Destructor to clear GPU Memory.

                    acc_run += accuracy
                    acc_run_list.append(accuracy)
                    INFO_LOGGER.info(f"Test Accuracy in run {i + 1} and split {j + 1}: {accuracy}")

                acc_run /= 10
                std_run = np.std(acc_run_list)
                acc_noise += acc_run
                std_noise += std_run

                start_time = time.time()

                gc.collect()
                torch.cuda.empty_cache()  # Clearing CUDA Cache after 10 Splits
                INFO_LOGGER.info("CUDA Memory Summary:")
                INFO_LOGGER.info(torch.cuda.memory_summary(device=None, abbreviated=False))
                INFO_LOGGER.info("Cleared CUDA Cache.")
                INFO_LOGGER.info(f"Time for CUDA Clearing: {time.time() - start_time}s")
            INFO_LOGGER.info(f"Test Accuracy over {self.hp.runs} runs: {acc_noise / self.hp.runs}")
            accuracy_list.append(acc_noise / self.hp.runs)
            std_list.append(std_noise / self.hp.runs)

        return accuracy_list, std_list

    def optimize_hp(self):
        INFO_LOGGER.info(f"Optimizing HP for Experiment with Model {self.model} and Dataset {self.dataset}")
        hp_yamls = os.listdir(os.path.join(PATH, "output/hpyamls"))

        for hp_yaml in hp_yamls:
            if f"{self.model}_{self.dataset.replace('-', '_').replace(' ', '_')}" in hp_yaml:
                try:
                    shutil.copy(os.path.join(PATH, f"output/hpyamls/{hp_yaml}"),
                                os.path.join(PATH, f"output/hpyamls/hp_{self.model}_"
                                                   f"{self.dataset.replace('-', '_').replace(' ', '_')}_"
                                                   f"{self.noise.replace('-', '_').replace(' ', '_')}.yaml"))
                except shutil.SameFileError:
                    pass
                INFO_LOGGER.info(f"HP already optimized for Model {self.model} and Dataset {self.dataset}")
                return

        best_hp = None
        if self.model == "gcn":
            best_hp = gcn.optimize_hp(self.dataset, self.dataset_obj, self.hp, self.noise)
        elif self.model == "sam":
            best_hp = gcn.optimize_hp_with_sam(self.dataset, self.dataset_obj, self.hp, self.noise)
        elif self.model == "nrgnn":
            data = load_data_split(self.dataset, None, None, 1)
            graph = to_networkx(data, node_attrs=["x"])
            adj = nx.adjacency_matrix(graph)
            best_hp = nrgnn_file.optimize_hp(self.dataset, self.dataset_obj, self.hp, self.noise, graph, adj)
        elif self.model == "lpm":
            data = load_data_split(self.dataset, None, None, 1)
            graph = to_networkx(data, node_attrs=["x"])
            adj = nx.adjacency_matrix(graph)
            num_classes = data.y.max().item() + 1
            y = np.array([[1 if i == label else 0 for i in range(num_classes)] for label in data.y])
            features = sp.csr_matrix(data.x.cpu().numpy())
            best_hp = lpm_file.optimize_hp(self.dataset, self.dataset_obj, self.hp, self.noise, adj, features,
                                           num_classes, y)
        elif self.model == "pown":
            best_hp = pown.optimize_hp(self.dataset, self.dataset_obj, self.hp, self.noise)
        else:
            ERROR_LOGGER.error(f"HP Optimization not possible for {self.model}")

        final_hp_dict = {"dataset": {"name": self.dataset}, "model": {},
                         "noise": {"name": self.noise, "type": "pair" if "pair" in self.noise else "uniform",
                                   "ratios": self.noise_ratios}}
        hp_dict = best_hp.copy()
        final_hp_dict["model"] = hp_dict
        file_path = (f"{PATH}/output/hpyamls/hp_{self.model}_{self.dataset.replace('-', '_')}_"
                     f"{self.noise.replace(' ', '_').replace('-', '_')}.yaml")

        parse_hp_dict_to_yaml(final_hp_dict, file_path)
        INFO_LOGGER.info(f"HP Optimization Completed")
        INFO_LOGGER.info(f"Created YAML File for Experiment: {self.model}, {self.dataset}, {self.noise}")
