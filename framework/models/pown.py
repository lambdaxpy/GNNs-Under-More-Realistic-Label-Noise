import itertools
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
from ml_collections.config_dict import config_dict
from torch_geometric.data import Dataset
from torch_geometric.nn import LabelPropagation

from framework.configparser.yamlparser import parse_yaml_file_to_config_dict
from framework.datasets.loader import load_data_split
from framework.env import PATH
from framework.log.logger import get_info_logger
from framework.models.gcn import Model
from framework.models.utils.mapper import DATASET, OPTIMIZER
from framework.noise.uniform import UniformNoise
from framework.resultbuilder.csvbuilder import parse_df_into_csv
from framework.resultbuilder.dfbuilder import build_df_from_dict
from framework.train.utils.earlystopping import EarlyStopping

INFO_LOGGER = get_info_logger(name=__name__)


def get_edge_weights(features, edge_index, prototypes_tensor, eps):
    # compute edge weights based
    src = edge_index[0, :]
    dst = edge_index[1, :]

    edge_weights_nodes = torch.norm(F.normalize(features[src, :]) - F.normalize(features[dst, :]), p=2.0, dim=1)
    edge_proto_dist = torch.cdist(F.normalize(features)[src, :], F.normalize(prototypes_tensor), p=2.0)

    edge_weight_proto = edge_proto_dist.min(dim=1).values
    edge_weights = 1 / (edge_weight_proto + edge_weights_nodes + eps)  # edge_weights_nodes +

    return edge_weights


def pseudo_label_lp(features, edge_index, prototypes_tensor, eps, data, entropy_max, alpha, u):
    edge_weights = get_edge_weights(features, edge_index, prototypes_tensor, eps)

    # pseudo labels for novel samples
    lp_mask = data.train_mask
    label_mask, pseudo_probs = pseudo_label_lp_step(edge_weights, lp_mask, data, 2, entropy_max, alpha, u)

    pseudo_labels = pseudo_probs.max(dim=1).indices

    return label_mask, pseudo_labels


def pseudo_label_lp_step(edge_weights, lp_mask, data, lp_hop, entropy_max, alpha, u):
    # Init label propagation
    lp = LabelPropagation(num_layers=lp_hop, alpha=alpha)
    pseudo_labels = data.y.detach().clone()
    # pseudo_label_mask = torch.zeros_like(data.y, dtype=torch.bool)

    # performing lp with seed points = data.train_mask
    # lp_mask = pseudo_label_mask & lp_mask

    # performs one lp
    pseudo_probs = lp(pseudo_labels, data.edge_index, lp_mask, edge_weight=edge_weights)

    pseudo_probs_sum = torch.sum(pseudo_probs, dim=1)
    pseudo_probs_mask = pseudo_probs_sum > 0  # Filter only nodes that are affected of the label propagation
    idx_pseudo_probs = torch.where(pseudo_probs_mask)[0]

    # get uncertainty
    temperature_lp = 10

    pseudo_probs = F.softmax(pseudo_probs * temperature_lp, dim=1)

    entropies_lp = -torch.sum(pseudo_probs * torch.log2(pseudo_probs), dim=1) / entropy_max
    entropy_weight = 0.2
    entropies = entropy_weight * entropies_lp
    entropies = entropies[pseudo_probs_mask]  # Filter only nodes that are affected of the label propagation
    _, indices = torch.sort(entropies, descending=True)

    num_lp_nodes = int(pseudo_probs_mask.sum())
    num_remove_samples = int(num_lp_nodes * u)

    idx_filtered = indices[num_remove_samples:]
    idx_correct_labeled = idx_pseudo_probs[idx_filtered]
    label_mask = torch.BoolTensor(
        [True if i in idx_correct_labeled else False for i in range(data.train_mask.shape[0])])

    return label_mask, pseudo_probs


class OpenGraph(torch.nn.Module):
    def __init__(self, encoder, hidden_channels, num_protos, known_classes, unknown_classes,
                 device, sup_loss_weight, pseudo_loss_weight, unsup_loss_weight, entropy_loss_weight,
                 geometric_loss_weight,
                 ood_percentile, lp_hop=1, lp_min_seeds=3, entropy_threshold=0.1, sup_temp=0.1, pseudo_temp=0.7,
                 log_all=True, proto_type="param", pseudo_label_method="none", alpha=0.9, u=0.6):

        super().__init__()

        super(OpenGraph, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_protos = num_protos

        self.known_classes = known_classes.to(device)
        self.unknown_classes = unknown_classes.to(device)
        # self.unlabeled_protos = [i for i in range(num_protos) if i not in known_classes]
        self.unlabeled_protos = torch.tensor([i for i in range(num_protos) if i not in known_classes]).to(device)
        self.offset_array = self.calc_offset_array(torch.arange(num_protos), unknown_classes).to(device)

        self.sup_temp = sup_temp
        self.pseudo_temp = pseudo_temp
        self.lp_hop = lp_hop
        self.lp_min_seeds = lp_min_seeds
        self.entropy_threshold = entropy_threshold

        uniform_class_distr = torch.full((self.num_protos,), 1 / self.num_protos, device=device)
        self.entropy_max = -torch.sum(uniform_class_distr * torch.log2(uniform_class_distr), dim=0).to(device)

        self.ood_percentile = ood_percentile
        self.pseudo_label_method = pseudo_label_method

        self.alpha = alpha
        self.u = u

        self.sup_loss_weight = sup_loss_weight
        self.pseudo_loss_weight = pseudo_loss_weight
        self.unsup_loss_weight = unsup_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.geometric_loss_weight = geometric_loss_weight

        self.encoder = encoder

        self.proto_type = proto_type

        self.device = device
        self.eps = 1e-9
        self.log_all = log_all

        self.prototypes = torch.nn.ModuleList()

        for i in range(num_protos):
            proto_i = ProtoRepre(hidden_channels, proto_type, device)
            self.prototypes.append(proto_i)

            # Gramm Schmidt
            # gs_v = gs(torch.stack(self.prototypes).numpy(), row_vecs=True, norm = True)
            # self.prototypes = [torch.tensor(gs_v[i], dtype=torch.float32).to(device) for i in range(num_protos)]

    def calc_offset_array(self, known_classes, unknown_classes):
        result = [torch.sum(elem > unknown_classes) for elem in known_classes]
        return torch.Tensor(result).type(torch.long)

    def forward(self, x, edge_index):
        features = self.encoder(x, edge_index)

        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device)
        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)

        probas = F.softmax(-distances, dim=1)
        return probas

    def inference(self, x, edge_index):

        features = self.encoder(x, edge_index)

        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(
            self.device)  # move init, check copy, extra method to return embeddin
        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)

        probas = F.softmax(-distances, dim=1)

        return probas.argmax(dim=1)

    # def get_embeddings(self, x, edge_index):
    # features = self.encoder(x, edge_index)
    # return features

    def train_one_epoch(self, optimizer, data):

        '''
        For Badro:

        This should do:
            - Calculate embeddings(here features) with your encoder
            - Calculate pseudo labels with label propagation code, remove most uncertain ones
            - Apply loss on prototypes

        '''

        optimizer.zero_grad()

        features = self.encoder(data.x, data.edge_index)
        label_mask, pseudo_labels = self.gen_pseudo_labels(features, data.edge_index, data)

        data.labeled_mask = label_mask  # uncertain labels removed here from training

        # print(torch.count_nonzero(data.labeled_mask))

        open_loss = self.open_loss(features, data, pseudo_labels)
        open_loss.backward(retain_graph=True)

        # print("Loss:", open_loss.item())

        optimizer.step()
        optimizer.zero_grad()

        return open_loss

    def open_loss(self, features, data, pseudo_labels):

        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device)

        # unsupervised loss -> DGI
        # unsup_loss = self.unsupervised_loss(data.x, data.edge_index, data.unlabeled_mask)

        # regularization to get an equal prototype distribution
        entropy_loss = self.entropy_regularizer(features, prototypes_tensor, data)

        # supervised loss
        # sup_loss = self.supervised_loss(features[data.labeled_mask], data.y[data.labeled_mask], self.sup_temp)
        sup_loss = self.proto_loss(features[data.labeled_mask],
                                   prototypes_tensor,
                                   pseudo_labels[data.labeled_mask],
                                   self.sup_temp)

        # open_loss = self.sup_loss_weight * sup_loss + self.entropy_loss_weight * entropy_loss
        # open_loss += self.geometric_loss_weight * (
        # self.class_seperability_loss(prototypes_tensor) + self.class_uniformity_loss(prototypes_tensor))
        return sup_loss * self.sup_loss_weight

    def proto_loss(self, features, prototypes_tensor, y, temperature):

        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)
        distances = torch.div(distances, temperature)

        probabilities_for_training = torch.nn.Softmax(dim=1)(-distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), y]

        loss = -torch.log(probabilities_at_targets).mean()

        return loss

    def supervised_loss(self, features, target, temperature):
        cosine_dist = features @ features.t()
        cosine_mat = torch.div(cosine_dist, temperature)
        mat_max, _ = torch.max(cosine_mat, dim=1, keepdim=True)

        cosine_mat = cosine_mat - mat_max.detach()

        target_ = target.contiguous().view(-1, 1)
        mask_pos = torch.eq(target_, target_.T)

        mask_neg_base = 1 - torch.diag(torch.ones(features.size(0))).to(self.device)

        pos_term = (cosine_mat * mask_pos).sum(1) / (mask_pos.sum(1) + self.eps)
        neg_term = (torch.exp(cosine_mat) * mask_neg_base).sum(1)

        log_term = (pos_term - torch.log(neg_term + self.eps))
        return -log_term.mean()

    # def unsupervised_loss(self, x, edge_index, unlabeled_mask):
    # pos_z, neg_z, summary = self.ssl.forward(x, edge_index, mask=unlabeled_mask)

    # loss = self.ssl.loss(pos_z, neg_z, summary)

    # return loss

    def class_seperability_loss(self, prototypes_tensor):
        distances = torch.cdist(F.normalize(prototypes_tensor), F.normalize(prototypes_tensor), p=2.0)
        neg_max_dists = torch.max(-distances, dim=1).values
        loss = torch.exp(neg_max_dists)

        return torch.mean(loss)

    def class_uniformity_loss(self, prototypes_tensor):
        proto_center = torch.mean(prototypes_tensor, dim=0).unsqueeze(0)

        normalized_proto = F.normalize(prototypes_tensor - proto_center)

        cos_dist = normalized_proto @ normalized_proto.t()
        unit_matrix = torch.eye(cos_dist.shape[0]).to(self.device)
        cos_dist = cos_dist - unit_matrix

        loss = torch.max(cos_dist, 1).values

        return torch.mean(loss)

    def entropy_regularizer(self, features, prototypes_tensor, data):

        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)

        # Assumption: Label distribution of test/train is similar - to drop this assumption use uniform distr.
        labeled_classes, count_labels = torch.unique(data.y[data.labeled_mask], return_counts=True)
        n_remaining_protos = self.num_protos - labeled_classes.size(0)

        # set minus prorotypes, labeled classes
        combined = torch.cat((torch.arange(start=0, end=self.num_protos).to(self.device), labeled_classes))
        uniques, counts = combined.unique(return_counts=True)
        unlabeled_classes = uniques[counts == 1].type(torch.LongTensor)

        labeled_proto_ratio = labeled_classes.size(0) / self.num_protos

        prior = torch.zeros(self.num_protos).to(self.device)

        if labeled_classes.size(0) == 0:
            prior[unlabeled_classes] = (1 / unlabeled_classes.size(0)) * (1 - labeled_proto_ratio)
        elif unlabeled_classes.size(0) == 0:
            prior[labeled_classes] = (count_labels / count_labels.sum(0)) * labeled_proto_ratio
        else:
            prior[labeled_classes] = (count_labels / count_labels.sum(0)) * labeled_proto_ratio
            prior[unlabeled_classes] = (1 / unlabeled_classes.size(0)) * (1 - labeled_proto_ratio)

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        probas = F.log_softmax(-distances, dim=1).max()

        loss = kl_loss(probas, prior)

        return distances.mean()

    def select_threshold(self, closest, percentile):
        q = torch.quantile(closest, percentile)
        return q

    def everything_to_device(self, device):
        self = self.to(device)
        self.known_classes = self.known_classes.to(device)
        self.unknown_classes = self.unknown_classes.to(device)
        self.unlabeled_protos = self.unlabeled_protos.to(device)
        self.offset_array = self.offset_array.to(device)
        self.entropy_max = self.entropy_max.to(device)
        self.device = device
        return self

    def gen_pseudo_labels(self, features, edge_index, data):
        # pseudo label all data points as id that are close to a known class prototype

        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device)

        label_mask, pseudo_labels = pseudo_label_lp(features=features,
                                                    edge_index=edge_index,
                                                    prototypes_tensor=prototypes_tensor,
                                                    # num_protos=self.num_protos,
                                                    # proto_subset=self.unlabeled_protos,
                                                    # known_classes=self.known_classes,
                                                    # ood_percentile=self.ood_percentile,
                                                    # lp_hop=self.lp_hop,
                                                    # entropy_threshold=self.entropy_threshold,
                                                    # entropy_max=self.entropy_max,
                                                    eps=self.eps,
                                                    data=data,
                                                    entropy_max=self.entropy_max,
                                                    alpha=self.alpha,
                                                    u=self.u)
        # device=self.device)

        return label_mask, pseudo_labels


class ProtoRepre(torch.nn.Module):
    def __init__(self, hidden_channels, proto_type, device):
        super(ProtoRepre, self).__init__()

        self.hidden_channels = hidden_channels
        self.proto_type = proto_type
        self.normalizer = lambda x: x / torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10

        self.device = device

        self.prototype = torch.Tensor(hidden_channels)
        torch.nn.init.normal_(self.prototype, mean=0.0, std=1.0)
        self.prototype = torch.nn.functional.normalize(self.prototype, p=2.0, dim=0)
        self.prototype = torch.nn.Parameter(self.prototype).to(device)

    # def update_proto(self, features, weights, y, label):
    # if self.proto_type == "mean":
    # class_wise_features = features[y == label, :]
    # weights = F.softmax(weights[y == label], dim=0).unsqueeze(1).expand_as(class_wise_features)
    # res = torch.matmul(class_wise_features, weights)
    # self.prototypes = self.normalizer(torch.sum(class_wise_features * weights, dim=0).to(self.device))
    # else:
    # raise NotImplementedError("Your prototype update is not defined!")

    # self.gat_layer = GATConv(in_channels=hidden_state, out_channels=hidden_state, heads=1, concat=False) #paper variant?
    # self.atten = nn.MultiheadAttention(hidden_state, 2) #code variant

    def forward(self):  # , spt_embedding_i, degree_list_i=None):
        return self.prototype


def evaluate_pown(dataset: Dataset, data: torch_geometric.data.Data, mask: torch.Tensor, hp_optimizer: str,
                  lr: float = 0.001,
                  weight_decay: float = 0.01, hidden_dim: int = 256, dropout: float = 0.0, norm: str = "none",
                  layer: int = 2, res: bool = False, epochs: int = 1000, patience: int = 50, alpha: float = 0.9,
                  u: float = 0.6, sup_loss_weight: float = 0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn = Model("gcn", layer, data.num_node_features, hidden_dim, dataset.num_classes,
                0, 0, norm, dropout, res)
    labels = data.y
    known_classes = torch.Tensor(list(range(labels.max().item() + 1)))
    pown = OpenGraph(gcn, dataset.num_classes, dataset.num_classes, known_classes, torch.Tensor([]), device,
                     sup_loss_weight, 0.1, 0, 0.1,
                     0.1, 5, alpha=alpha, u=u)
    gcn.to(device)
    pown.to(device)
    pown.everything_to_device(device)

    INFO_LOGGER.info(f"Evaluating POWN on {device}")

    timestamp = time.time()
    optimizer = OPTIMIZER[hp_optimizer](pown.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience, Path(os.path.join(PATH, f"train/esmodels/pown_{str(timestamp)}.pt")))
    stopped_early = False

    for epoch in range(epochs):
        pown.train()
        pown.train_one_epoch(optimizer, data)

        pown.eval()
        pred = pown(data.x, data.edge_index)
        pred = pred.argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        val_acc = int(correct) / int(data.x[data.val_mask].size()[0])

        if early_stopping(val_acc, pown):
            INFO_LOGGER.info(f"Early Stopping at Epoch: {epoch + 1}")
            stopped_early = True
            break

    if stopped_early:
        pown.load_state_dict(torch.load(os.path.join(PATH, f"train/esmodels/pown_{dataset.__str__()}.pt")))

    pown.eval()
    pred = pown(data.x, data.edge_index)
    pred = pred.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    accuracy = int(correct) / int(data.x[mask].size()[0])

    return accuracy


def optimize_hp(dataset: str, dataset_obj: torch_geometric.data.Dataset, hp: config_dict.ConfigDict, noise_type: str) \
        -> dict:
    INFO_LOGGER.info("Starting HP Optimization of POWN")

    hp_optimizer = hp.optimizer
    lr = hp.lr
    patience = hp.patience
    weight_decay = hp["weight-decay"]
    hidden_dim = hp["hidden_dim"]
    dropout = hp.dropout
    norm = hp.norm
    layer = hp.layers
    res = hp.res
    epochs = hp.epochs
    sup_loss_weights = hp.sup_loss_weight
    alphas = hp.alpha
    u_s = hp.u

    hp_dict = hp.to_dict()

    df_dict = {"split": [], "sup_loss_weight": [], "alpha": [],
               "u": [], "accuracy": []}

    best_hp_dict = hp_dict.copy()
    best_accuracy = -1.0

    cartesian_product = itertools.product(sup_loss_weights, alphas, u_s)

    for (sup_loss_weight, alpha, u) in cartesian_product:
        INFO_LOGGER.info(f"Training for: ({sup_loss_weight}, {alpha}, {u})")
        acc_sum = 0.
        for i in range(10):  # number of splits
            INFO_LOGGER.info(f"HP Optimization of POWN with: {i + 1}. Split")
            data = load_data_split(dataset, None, None, i + 1)
            INFO_LOGGER.info(f"Is Undirected: {data.is_undirected()}")
            accuracy = evaluate_pown(dataset_obj, data, data.val_mask, hp_optimizer, lr, weight_decay, hidden_dim,
                                     dropout,
                                     norm, layer, res, epochs, patience, alpha, u)
            INFO_LOGGER.info(f"Accuracy: {accuracy}")
            acc_sum += accuracy
            df_dict["sup_loss_weight"].append(sup_loss_weight)
            df_dict["alpha"].append(alpha)
            df_dict["u"].append(u)
            df_dict["accuracy"].append(accuracy)
            df_dict["split"].append(i + 1)

        avg_acc = acc_sum / 10
        if avg_acc > best_accuracy:
            best_hp_dict["sup_loss_weight"] = sup_loss_weight
            best_hp_dict["alpha"] = alpha
            best_hp_dict["u"] = u
            best_accuracy = avg_acc

        INFO_LOGGER.info(f"POWN Average Validation Accuracy: {avg_acc}")

    df = build_df_from_dict(df_dict)
    parse_df_into_csv(df, os.path.join(PATH, f"output/hpresults/pown_{dataset}_{noise_type}.csv"))
    INFO_LOGGER.info(f"Best HP for POWN: {best_hp_dict}")

    return best_hp_dict


if __name__ == '__main__':
    config = parse_yaml_file_to_config_dict(os.path.join(PATH, "tests/testconfig.yaml"))
    dataset_obj = DATASET["cora"](root="../datasets", name="cora", split="random", num_train_per_class=20, num_val=500,
                                  num_test=1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset_obj[0].to(device)
    labels = data.y
    known_classes = torch.Tensor(list(range(labels.max().item() + 1)))

    print(evaluate_pown(dataset_obj, data, data.val_mask, "adam"))
