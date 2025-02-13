import pandas as pd

df = pd.read_csv("pown_pubmed_feature-based pair.csv")
df = df[df["u"] == 0.5]
df = df[df["alpha"] == 0.2]
print(df.groupby("sup_loss_weight")["accuracy"].mean())