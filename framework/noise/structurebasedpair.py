import torch
import torch_geometric.data
from torch_geometric.data import Data
from framework.noise.pair import PairNoise
from torch_geometric.utils import k_hop_subgraph


class StructureBasedPairNoise(PairNoise):
    def __init__(self, data: Data, num_of_classes: int) -> None:
        super().__init__(data, num_of_classes)

    def add_noise(self, noise_ratio: float) -> torch_geometric.data.Data:
        class_pairs = self.determine_class_pairs()
        return super().add_noise(noise_ratio, class_pairs)

    def determine_class_pairs(self) -> [(int, int)]:
        class_pairs = []
        for i in range(self.num_of_classes):
            class_vector = self.get_neighboring_class_vector(i)
            similar_class = torch.argmax(class_vector)
            class_pairs.append((i, similar_class))

        return class_pairs

    def get_neighboring_class_vector(self, label: int) -> torch.Tensor:
        class_vector = [0 for _ in range(self.num_of_classes)]
        for i in range(self.data.x.size()[0]):
            if self.data.y[i] == label:
                neighbors, _, _, _ = k_hop_subgraph(i, 1, self.data.edge_index)
                for neighbor in neighbors:
                    if neighbor != i and self.data.y[neighbor] != label:
                        neighbor_label = self.data.y[int(neighbor)]
                        class_vector[int(neighbor_label)] += 1

        return torch.Tensor(class_vector)


if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    y = torch.tensor([1, 1, 0], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.t().contiguous())
    structure_based_pair = StructureBasedPairNoise(data, num_of_classes=2)
    print(structure_based_pair.get_neighboring_class_vector(1))
