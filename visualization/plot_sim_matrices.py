import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch

from framework.env import PATH
from framework.resultbuilder.csvbuilder import parse_df_into_csv
from framework.datasets.loader import load_data_split
from framework.noise.featurebasedpair import FeatureBasedPairNoise
from framework.noise.structurebasedpair import StructureBasedPairNoise


def plot_sim_matrices(dataset: str):
    data = load_data_split(dataset, None, None, 1).cpu()

    fb_sim_matrix = np.zeros((data.y.max().item() + 1, data.y.max().item() + 1), dtype=float)
    sb_sim_matrix = fb_sim_matrix.copy()
    noise = FeatureBasedPairNoise(data, data.y.max() + 1)
    for i in range(data.y.max().item() + 1):
        for j in range(data.y.max().item() + 1):
            if i == j:
                fb_sim_matrix[i, j] = 0
                continue
            feature_set_i = noise.get_feature_set_of_class(i)
            feature_set_j = noise.get_feature_set_of_class(j)
            class_sim = noise.calculate_class_similarity(feature_set_i, feature_set_j)
            fb_sim_matrix[i, j] = class_sim

    noise = StructureBasedPairNoise(data, data.y.max() + 1)
    for i in range(data.y.max().item() + 1):
        for j in range(data.y.max().item() + 1):
            class_vector = noise.get_neighboring_class_vector(i)
            class_repr = class_vector[j]
            class_sim = class_repr / torch.sum(class_vector)
            sb_sim_matrix[i, j] = class_sim

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
    axs[0].title.set_text("Feature-based Similarity")
    axs[1].title.set_text("Structure-based Similarity")
    annot = True
    if dataset == "roman-empire":
        annot = False
    sns.heatmap(fb_sim_matrix, annot=annot, fmt='.2f', cmap="crest", ax=axs[0])
    sns.heatmap(sb_sim_matrix, annot=annot, fmt='.2f', cmap="crest", ax=axs[1])
    fig.savefig(f"sim_matrices/heatmap_{dataset}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_sim_matrices("amazon-ratings")
