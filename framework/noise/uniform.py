import random

import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.utils import mask_to_index
from framework.noise.noise import Noise


class UniformNoise(Noise):
    def __init__(self, data: Data, num_of_classes: int) -> None:
        super().__init__("uniform", data, num_of_classes)

    def add_noise(self, noise_ratio: float) -> torch_geometric.data.Data:
        super().add_noise(noise_ratio)
        data = self.data.clone()
        number_of_noisy_labels = int(noise_ratio * data.train_mask.sum().item())
        index_list = mask_to_index(data.train_mask).tolist()
        for i in range(number_of_noisy_labels):
            random_index = index_list[random.randint(0, len(index_list) - 1)]
            current_label = data.y[random_index]
            data.y[random_index] = self.flip_label(int(current_label))
            index_list.remove(random_index)

        return data

    def flip_label(self, current_label: int) -> int:
        labels = [label for label in range(self.num_of_classes) if label != current_label]
        return labels[random.randint(0, len(labels) - 1)]
