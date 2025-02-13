# Beyond Uniform Noise: Evaluating Graph Neural Networks Under Realistic Label Noise

This is the official repository of the paper "Beyond Uniform Noise: Evaluating Graph Neural Networks Under Realistic Label Noise".


## Project Structure

The repository is divided into two components.

**Framework: (```./framework```)**
The Framework is a Machine Learning pipeline for the experiments and hyperparameter optimization.

**Experiment: (```./experiment```)**
The Experiment holds the YAML configuration files of all experiments for the Framework.
For reproducibility, we have stored our experiments in separate config folders.


## Requirements

Python version: 3.11.2 Dependencies in requirements.txt

## Run the scripts

To run the hyperparameter optimization with YAML configs for hyperparameter optimization, use:

```python3 hp_tuning.py <config_folder>```

To run any experiments with YAML configs for evaluation, use:

```python3 main.py <config_folder>```

To reproduce our results, just run ```python3 main.py <config_folder>``` with our prefilled config folders in ```./experiment```.


## Noise Types

The implementation of **uniform noise**, **feature-based pair noise**, and **structure-based pair noise** can be found in:

```./framework/noise```

## Datasets

We used the following datasets: **Cora**, **Citeseer**, **Pubmed**, **Roman-empire**, **Amazon-ratings**

**Cora**, **Citeseer**, and **Pubmed** are the standard ```Planetoid``` datasets in ```torch-geometric```

**Roman-empire**, and **Amazon-ratings** are instances of the ```HeterophilousDataset``` class of ```torch-geometric```

For each dataset, we computed 10 different splits with each noise type.

The splits are not delivered in this repo. If you want to reproduce the experiment, please download the splits from https://shorturl.at/gED7F.
Create a ```splits``` folder in ```./framework/models``` and paste the splits into the folder.

## Methods

We tested the following methods on all three noise types and datasets:

1. **GCN:**
From Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification
with Graph Convolutional Networks. 

    [GCN implementation](./framework/models/gcn.py)

2. **NRGNN:** From Enyan Dai, Charu Aggarwal, and Suhang Wang. 2021. NRGNN: Learning a
Label Noise Resistant Graph Neural Network on Sparsely and Noisily Labeled
Graphs.

    [NRGNN implementation](./framework/models/nrgnn.py)
    
    The implementation is fully taken from the repo: https://github.com/EnyanDai/NRGNN
3. **D-GNN**: From Hoang NT, Choong Jun Jin, and Tsuyoshi Murata. 2019. Learning Graph Neu-
ral Networks with Noisy Labels.

    [D-GNN implementation](./framework/models/dgnn.py)
    
    The implementation is taken from https://github.com/EnyanDai/NRGNN with an adjustment to our GCN backbone.

4. **LPM:** From Jun Xia, Haitao Lin, Yongjie Xu, Lirong Wu, Zhangyang Gao, Siyuan Li, and
Stan Z. Li. 2021. Towards Robust Graph Neural Networks against Label Noise.

    [LPM implementation](./framework/models/lpm.py)

    The implementation is fully taken from the supplementary material of the paper: https://openreview.net/forum?id=H38f_9b90BO

5. **POWN:** From Marcel Hoffmann, Lukas Galke, and Ansgar Scherp. 2024. POWN: Prototypical
Open-World Node Classification.

    [POWN implementation](./framework/models/pown.py)

    The implementation is taken from https://github.com/Bobowner/POWN but changed to fit the label noise use case.

6. **SAM:** From Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. 2021.
Sharpness-aware Minimization for Efficiently Improving Generalization.

    [SAM implementation](./framework/models/sam.py)
    
    The implementation is taken from https://github.com/davda54/sam but is integrated into ```./framework/models/gcn.py```
