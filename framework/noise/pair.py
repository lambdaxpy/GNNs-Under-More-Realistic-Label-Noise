import random

import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.utils import mask_to_index
from framework.noise.noise import Noise


class PairNoise(Noise):
    def __init__(self, data: Data, num_of_classes: int) -> None:
        super().__init__("pair", data, num_of_classes)

    def add_noise(self, noise_ratio: float, class_pairs: [(int, int)]) -> torch_geometric.data.Data:
        super().add_noise(noise_ratio)
        data = self.data.clone()
        number_of_noisy_labels = int(noise_ratio * data.train_mask.sum().item())
        index_list = mask_to_index(data.train_mask).tolist()
        for i in range(number_of_noisy_labels):
            random_index = index_list[random.randint(0, len(index_list) - 1)]
            current_label = data.y[random_index]
            data.y[random_index] = self.flip_label(int(current_label), class_pairs)
            index_list.remove(random_index)

        return data

    def flip_label(self, current_label: int, class_pairs: [(int, int)]) -> int:
        for class_pair in class_pairs:
            label_a, label_b = class_pair
            if label_a == current_label:
                return label_b
            elif label_b == current_label:
                return label_a

