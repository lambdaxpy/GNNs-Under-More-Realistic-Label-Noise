from torch_geometric.data import Data


class Noise:
    def __init__(self, type: str, data: Data, num_of_classes: int) -> None:
        self.type = type
        self.data = data
        self.num_of_classes = num_of_classes

    def add_noise(self, noise_ratio: float) -> None:
        if noise_ratio < 0 or noise_ratio > 1:
            raise ValueError('noise_ratio must be between 0 and 1')
