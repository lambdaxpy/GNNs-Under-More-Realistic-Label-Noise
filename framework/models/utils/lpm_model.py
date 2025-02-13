import torch
from torch import nn
from torch.nn import functional as F
from framework.models.utils.lpm_layer import GraphConvolution
import numpy as np


class GCN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = GraphConvolution(input_dim, hid_dim, num_features_nonzero,
                                       dropout=0.5,
                                       is_sparse_inputs=True)

        self.layer2 = GraphConvolution(hid_dim, output_dim, num_features_nonzero,
                                       dropout=0.5,
                                       is_sparse_inputs=False)

    def forward(self, inputs):
        x, support = inputs
        x_hidden, support = self.layer1((x, support))
        x_hidden = F.relu(x_hidden)
        x, support = self.layer2((x_hidden, support))
        return x, x_hidden


class ANet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(ANet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

