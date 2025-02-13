import torch
from unittest import TestCase
from framework.noise.pair import PairNoise
from torch_geometric.data import Data


class TestUniformNoise(TestCase):
    def setUp(self):
        self.num_classes = 6
        self.data = Data(x=torch.randn(10, 3), y=torch.randint(1, 6, (10,)))
        self.pair_noise = PairNoise(self.data, self.num_classes)

    def test_add_noise(self):
        self.assertNotEqual(self.pair_noise.add_noise(0.2, [(i, i + 1) for i in range(1, self.num_classes, 2)]), self.data)
