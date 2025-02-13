import os
import torch
import torch_geometric
from framework.log.logger import get_info_logger, get_error_logger
from framework.env import PATH


INFO_LOGGER = get_info_logger(name=__name__)
ERROR_LOGGER = get_error_logger(name=__name__)


def load_data_split(dataset: str, noise: str | None, noise_ratio: float | None, split: int) -> torch_geometric.data.Data:
    INFO_LOGGER.info(f"Loading {split}. Data Split from {dataset} with Noise: {noise} and Noise Ratio: {noise_ratio}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset not in ["citeseer", "cora", "pubmed", "roman-empire", "amazon-ratings", "photo"]:
        ERROR_LOGGER.error(f"Dataset {dataset} not present.")
        raise ValueError(f"Dataset {dataset} not present.")

    if split < 1 or split > 10:
        ERROR_LOGGER.error(f"Split with number {split} does not exist.")
        raise ValueError(f"Split with number {split} does not exist.")

    if noise not in [None, "uniform", "feature-based pair", "structure-based pair"]:
        ERROR_LOGGER.error(f"Noise {noise} does not exist.")
        raise ValueError(f"Noise {noise} does not exist.")

    if noise is None:
        data_file_path = os.path.join(PATH, f"datasets/splits/{dataset}/standard_{split}.pt")
        data = torch.load(data_file_path)
        data.to(device)
    else:
        noise = noise.replace("-", "_").replace(" ", "_")
        if noise_ratio is None:
            ERROR_LOGGER.error("Noise ratio not set.")
            raise ValueError("Noise ratio not set.")
        noise_ratio_str = str(noise_ratio).replace(".", "_")
        data_file_path = os.path.join(PATH, f"datasets/splits/{dataset}/{noise}_{noise_ratio_str}_{split}.pt")
        data = torch.load(data_file_path)
        data.to(device)
    INFO_LOGGER.info(f"Data Split was loaded with {device}.")
    return data
