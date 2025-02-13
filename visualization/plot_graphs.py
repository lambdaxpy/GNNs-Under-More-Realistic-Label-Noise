import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from framework.env import PATH
from framework.resultbuilder.csvbuilder import parse_df_into_csv


def concat_csv_files(folder_path: str):
    all_files = os.listdir(folder_path)
    li = []
    for filename in all_files:
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename), index_col=None, header=0)
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    parse_df_into_csv(frame, os.path.join(folder_path, 'results.csv'))


def plot_dgnn_test(datasets: list[str], folder_path: str):
    noise_types = ["uniform", "feature-based pair", "structure-based pair"]
    df = pd.read_csv(os.path.join(folder_path, "results.csv"))

    fig, axs = plt.subplots(nrows=len(datasets), ncols=3)
    for i, dataset in enumerate(datasets):
        for j, noise_type in enumerate(noise_types):
            d_df = df[df["dataset"] == dataset]
            nt_df = d_df[d_df["noise_type"] == noise_type]
            nt_df = nt_df.drop(columns=["noise_type", "dataset", "model", "std"])
            nt_df = nt_df[nt_df["noise_ratio"] != 0.0]
            sns.set_theme()
            sns.lineplot(nt_df, x="noise_t", y="accuracy", hue="noise_ratio", ax=axs[i, j])
            print(nt_df)
            axs[i, j].set_xlabel("Noise Prior")
            axs[i, j].set_ylabel("Accuracy")
            axs[i, j].legend(title="Noise Ratio")
            if i != 0 or j != 0:
                axs[i, j].get_legend().remove()

            axs[i, j].title.set_text(f"{noise_type} noise")
            axs[i, j].grid()

    fig.set_figwidth(15)
    fig.set_figheight(10)
    plt.savefig(f"dgnn_plots/plot_{"_".join(datasets)}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_graphs(datasets: list[str], models: list[str]):
    noise_types = ["uniform", "feature-based pair", "structure-based pair"]
    name_map = {"gcn": "GCN", "nrgnn": "NRGNN", "lpm": "LPM", "dgnn": "D-GNN", "pown": "POWN", "sam": "SAM"}

    fig, axs = plt.subplots(nrows=len(datasets), ncols=3)
    for i, dataset in enumerate(datasets):
        for j, noise_type in enumerate(noise_types):
            model_accuracies = {}
            model_accuracies["noise_ratio"] = [0, 0.2, 0.4, 0.6, 0.8]
            for model in models:
                csv_file = os.path.join(PATH, f"output/results/{model}/results.csv")
                df = pd.read_csv(csv_file)
                filtered_df = df[df["noise_type"] == noise_type]
                filtered_df = filtered_df[filtered_df["dataset"] == dataset]
                accuracies = filtered_df["accuracy"].tolist()
                if noise_type != "uniform":
                    uniform_df = df[df["noise_type"] == "uniform"]
                    uniform_df = uniform_df[uniform_df["dataset"] == dataset]
                    original_acc = [uniform_df["accuracy"].tolist()[0]]
                    accuracies = original_acc + accuracies

                model_accuracies[name_map[model]] = accuracies

            sns.set_theme()
            print(model_accuracies)
            df = pd.DataFrame.from_dict(model_accuracies)
            print(pd.melt(df, ["noise_ratio"]))
            if len(datasets) > 1:
                sns.lineplot(data=pd.melt(df, ["noise_ratio"]), x="noise_ratio", y="value", hue="variable",
                             ax=axs[i, j])
                axs[i, j].set_xlabel("Noise Ratio")
                axs[i, j].set_ylabel("Accuracy")
                axs[i, j].legend(title="Model")
                if i != 0 or j != 0:
                    axs[i, j].get_legend().remove()
            else:
                sns.lineplot(data=pd.melt(df, ["noise_ratio"]), x="noise_ratio", y="value", hue="variable",
                             ax=axs[j])
                axs[j].set_xlabel("Noise Ratio")
                axs[j].set_ylabel("Accuracy")
                axs[j].legend(title="Model")
                if j != 0:
                    axs[j].get_legend().remove()

            if len(datasets) > 1:
                axs[i, j].title.set_text(f"{noise_type} noise")
                axs[i, j].grid()
            else:
                axs[j].title.set_text(f"{noise_type} noise")
                axs[j].grid()

    fig.set_figwidth(15)
    fig.set_figheight(5)
    plt.savefig(f"plots/plot_{"_".join(datasets)}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # concat_csv_files(os.path.join(PATH, "output/results/pown"))
    # plot_graphs(["amazon-ratings"], ["gcn", "dgnn", "sam", "lpm", "nrgnn", "pown"])
    plot_dgnn_test(["cora", "roman-empire"], os.path.join(PATH, "output/results/dgnn_test"))
