import torch
from unittest import TestCase
from torch_geometric.data import Data
from framework.noise.featurebasedpair import FeatureBasedPairNoise


class TestFeatureBasedPairNoise(TestCase):
    def setUp(self):
        self.num_classes = 2
        self.data = Data(x=torch.Tensor([[1, 2, 4], [4, 1, 2]]), y=torch.Tensor([0, 1]))
        self.data.train_mask = torch.tensor([True, True, True])
        self.pair_noise = FeatureBasedPairNoise(self.data, self.num_classes)

    def test_get_feature_set_of_classes(self):
        vec_list = self.pair_noise.get_feature_set_of_class(1)
        self.assertEqual(len(vec_list), 1)

    def test_determine_class_pairs(self):
        self.assertEqual(self.pair_noise.determine_class_pairs(), [(0, 1), (1, 0)])

    def test_add_noise(self):
        self.assertTrue(self.pair_noise.add_noise(0.5) != torch.Tensor([0, 1]))
