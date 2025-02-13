import torch
import os
import random

import torch_geometric.data
from torch_geometric.datasets import Planetoid, Amazon, HeterophilousGraphDataset
from framework.noise.uniform import UniformNoise
from framework.noise.featurebasedpair import FeatureBasedPairNoise
from framework.noise.structurebasedpair import StructureBasedPairNoise
from framework.log.logger import get_info_logger
from framework.env import PATH

INFO_LOGGER = get_info_logger(name=__name__)

NOISE_TYPES = ["uniform", "feature-based pair", "structure-based pair"]
NOISE_RATIOS = [0.2, 0.4, 0.6, 0.8]


def get_nodes_of_class(data: torch_geometric.data.Data, label: int, ignore_nodes: list) -> torch.Tensor:
    desired_nodes = []
    for i in range(data.num_nodes):
        if data.y[i] == label and i not in ignore_nodes:
            desired_nodes.append(i)

    return torch.tensor(desired_nodes)


def get_random_nodes(nodes: torch.Tensor, amount: int) -> torch.Tensor:
    index_list = list(range(nodes.size(dim=0)))
    random_nodes = []
    for _ in range(amount):
        random_index = index_list[random.randint(0, len(index_list) - 1)]
        random_nodes.append(nodes[random_index])
        index_list.remove(random_index)

    return torch.tensor(random_nodes)


def split_photo(data: torch_geometric.data.Data, num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_nodes = torch.tensor([], dtype=torch.long)
    val_nodes = torch.tensor([], dtype=torch.long)
    test_nodes = torch.tensor([], dtype=torch.long)
    ignore_nodes = []
    for i in range(num_classes):
        nodes = get_nodes_of_class(data, i, ignore_nodes)
        random_nodes = get_random_nodes(nodes, 20)  # 20 random training nodes per class
        ignore_nodes += random_nodes
        train_nodes = torch.cat((train_nodes, random_nodes), dim=0)

    for i in range(num_classes):
        nodes = get_nodes_of_class(data, i, ignore_nodes)
        random_nodes = get_random_nodes(nodes, 30)  # 20 random validation nodes per class
        ignore_nodes += random_nodes
        val_nodes = torch.cat((val_nodes, random_nodes), dim=0)

    rest_nodes = [i for i in range(data.num_nodes) if i not in ignore_nodes]
    test_nodes = torch.cat((test_nodes, torch.tensor(rest_nodes)), dim=0)

    train_mask = torch.zeros(data.num_nodes)
    val_mask = torch.zeros(data.num_nodes)
    test_mask = torch.zeros(data.num_nodes)

    train_mask[train_nodes] = 1
    val_mask[val_nodes] = 1
    test_mask[test_nodes] = 1

    return train_mask, val_mask, test_mask


def download_splits(name: str):
    # without noise 10 splits
    num_classes = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(10):
        INFO_LOGGER.info(f"Without Noise. {i + 1}. Sample")
        if name in ["cora", "citeseer", "pubmed"]:
            dataset = Planetoid(root="./", name=name, split="random", num_train_per_class=20, num_val=500,
                             num_test=1000)
            data = dataset[0].to(device)
        elif name in ["amazon-ratings", "roman-empire"]:
            dataset = HeterophilousGraphDataset(root="./", name=name)
            data = dataset[0].to(device)
            data.train_mask = data.train_mask[:, i]
            data.val_mask = data.val_mask[:, i]
            data.test_mask = data.test_mask[:, i]
        elif name == "photo":
            dataset = Amazon(root="./", name=name)
            data = dataset[0].to(device)
            train_mask, val_mask, test_mask = split_photo(data, dataset.num_classes)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        num_classes = dataset.num_classes
        INFO_LOGGER.info(f"Saving Dataset Split into File.")
        torch.save(data, os.path.join(PATH, f"datasets/splits/{name}/standard_{i + 1}.pt"))

    # with every noise type and ratio 10 splits
    for noise_type in NOISE_TYPES:
        for noise_ratio in NOISE_RATIOS:
            for i in range(10):
                data = torch.load(os.path.join(PATH, f"datasets/splits/{name}/standard_{i + 1}.pt"))
                data.to(device)
                if noise_type == "uniform":
                    INFO_LOGGER.info(f"Uniform Noise with ratio: {noise_ratio}. {i + 1}. Sample")
                    noise = UniformNoise(data, num_classes)
                    noise_obj = noise.add_noise(noise_ratio)
                    INFO_LOGGER.info(f"Saving Dataset Split into File.")
                    torch.save(noise_obj, os.path.join(PATH,
                                                       f"datasets/splits/{name}/{noise_type}_"
                                                       f"{str(noise_ratio).replace('.', '_')}_{i + 1}.pt"))
                elif noise_type == "feature-based pair":
                    INFO_LOGGER.info(f"Feature-based Pair Noise with ratio: {noise_ratio}. {i + 1}. Sample")
                    noise = FeatureBasedPairNoise(data, num_classes)
                    noise_obj = noise.add_noise(noise_ratio)
                    INFO_LOGGER.info(f"Saving Dataset Split into File.")
                    torch.save(noise_obj, os.path.join(PATH,
                                                       f"datasets/splits/{name}/{noise_type.replace('-', '_').replace(' ', '_')}"
                                                       f"_{str(noise_ratio).replace('.', '_')}_{i + 1}.pt"))

                elif noise_type == "structure-based pair":
                    INFO_LOGGER.info(f"Structure-based Pair Noise with ratio: {noise_ratio}. {i + 1}. Sample")
                    noise = StructureBasedPairNoise(data, num_classes)
                    noise_obj = noise.add_noise(noise_ratio)
                    INFO_LOGGER.info(f"Saving Dataset Split into File.")
                    torch.save(noise_obj, os.path.join(PATH,
                                                       f"datasets/splits/{name}/{noise_type.replace('-', '_').replace(' ', '_')}"
                                                       f"_{str(noise_ratio).replace('.', '_')}_{i + 1}.pt"))


def download_cora():
    download_splits("cora")


def download_citeseer():
    download_splits("citeseer")


def download_pubmed():
    download_splits("pubmed")


def download_photo():
    download_splits("photo")


def download_roman_empire():
    download_splits("roman-empire")


def download_amazon_ratings():
    download_splits("amazon-ratings")


if __name__ == '__main__':
    # data_1 = torch.load(os.path.join(PATH, "datasets/splits/cora/standard_1.pt"))
    # data_2 = torch.load(os.path.join(PATH, "datasets/splits/cora/standard_2.pt"))
    # print(data_1.y[data_1.train_mask])
    # print(data_2.y[data_2.train_mask])
    # download_cora()
    # download_citeseer()
    # download_pubmed()
    # download_photo()
    download_roman_empire()
    # download_amazon_ratings()
