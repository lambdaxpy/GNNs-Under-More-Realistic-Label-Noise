import torch
from unittest import TestCase
from framework.noise.uniform import UniformNoise
from torch_geometric.data import Data


class TestUniformNoise(TestCase):
    def setUp(self):
        self.data = Data(x=torch.randn(10, 3), y=torch.randint(1, 6, (10,)))
        self.uniform_noise = UniformNoise(self.data, 6)

    def test_add_noise(self):
        self.assertNotEqual(self.uniform_noise.add_noise(0.2), self.data)
