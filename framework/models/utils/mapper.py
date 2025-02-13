import torch
from torch_geometric.datasets import Planetoid, Amazon, HeterophilousGraphDataset


OPTIMIZER = {"adam": torch.optim.Adam,
             "adamw": torch.optim.AdamW,
             "sgd": torch.optim.SGD,
             "rprop": torch.optim.Rprop}


DATASET = {"cora": Planetoid,
           "citeseer": Planetoid,
           "photo": Amazon,
           "pubmed": Planetoid,
           "roman-empire": HeterophilousGraphDataset,
           "amazon-ratings": HeterophilousGraphDataset}
