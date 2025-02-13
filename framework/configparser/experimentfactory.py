from ml_collections import config_dict
from framework.experiment.experiment import Experiment


def create_experiment(config: config_dict.ConfigDict) -> Experiment:
    dataset = config.dataset.name
    model = config.model.name
    noise = config.noise.name
    noise_ratios = config.noise.ratios
    hp = config.model
    return Experiment(dataset, model, noise, noise_ratios, hp, config)
