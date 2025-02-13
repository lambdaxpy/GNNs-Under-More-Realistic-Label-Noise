import torch
from torch import nn
from torch_geometric.nn import GCNConv


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, x, edge_index):
        x_res = self.normalization(x)
        x_res = self.module(x_res, edge_index)
        x = x + x_res

        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GCNModule(nn.Module):
    def __init__(self, dim, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.gcn_conv = GCNConv(in_channels=dim, out_channels=dim)

    def forward(self, x, edge_index):
        x = self.gcn_conv(x, edge_index)
        x = self.dropout(x)
        x = self.act(x)
        return x
