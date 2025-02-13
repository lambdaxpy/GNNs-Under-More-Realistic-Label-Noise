import torch
import torch_geometric.data
from torch_geometric.data import Data
from framework.noise.pair import PairNoise


class FeatureBasedPairNoise(PairNoise):
    def __init__(self, data: Data, num_of_classes: int) -> None:
        super().__init__(data, num_of_classes)

    def add_noise(self, noise_ratio: float) -> torch_geometric.data.Data:
        class_pairs = self.determine_class_pairs()
        return super().add_noise(noise_ratio, class_pairs)

    def determine_class_pairs(self) -> [(int, int)]:
        class_pairs = []
        for i in range(self.num_of_classes):
            similar_class = 0
            highest_similarity = 0
            for j in range(i + 1, self.num_of_classes):
                feature_set_i = self.get_feature_set_of_class(i)
                feature_set_j = self.get_feature_set_of_class(j)
                similarity = FeatureBasedPairNoise.calculate_class_similarity(feature_set_i, feature_set_j)
                if similarity > highest_similarity:
                    similar_class = j
            class_pairs.append((i, similar_class))
        return class_pairs

    def get_feature_set_of_class(self, label: int) -> [torch.Tensor]:
        feature_set = []
        for i, tensor in enumerate(self.data.x):
            if self.data.y[i] == label:
                feature_set.append(tensor)
        return feature_set

    @staticmethod
    def calculate_class_similarity(feature_set_i: torch.Tensor, feature_set_j: torch.Tensor) -> float:
        similarity = 0
        for vector_i in feature_set_i:
            for vector_j in feature_set_j:
                similarity += vector_i.dot(vector_j).data
        return similarity / (len(feature_set_i) * len(feature_set_j))




