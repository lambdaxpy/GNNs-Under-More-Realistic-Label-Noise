import os
import time

from framework.experiment.experiment import Experiment
from framework.log.logger import get_info_logger
from framework.resultbuilder.dfbuilder import build_df_from_dict
from framework.resultbuilder.csvbuilder import parse_df_into_csv
from framework.env import PATH


INFO_LOGGER = get_info_logger(name=__name__)


class ExperimentCollection:
    def __init__(self, experiments: [Experiment]):
        self.experiments = experiments

    def __repr__(self):
        return "\n\n".join([f"Experiment {i + 1}" + "\n" + experiment.__repr__()
                            for i, experiment in enumerate(self.experiments)])

    def add_experiment(self, experiment: Experiment):
        self.experiments.append(experiment)

    def evaluate_all_experiments(self, dgnn_test: bool):
        df_dict = {"model": [], "dataset": [], "noise_type": [], "noise_ratio": [], "accuracy": [], "std": []}
        if dgnn_test:
            df_dict["noise_t"] = []
        for experiment in self.experiments:
            accuracies, stds = experiment.evaluate_experiment()
            amount_entries = len(accuracies)
            df_dict["model"] = df_dict["model"] + [experiment.model for _ in range(amount_entries)]
            df_dict["dataset"] = df_dict["dataset"] + [experiment.dataset for _ in range(amount_entries)]
            df_dict["noise_type"] = df_dict["noise_type"] + [experiment.noise for _ in range(amount_entries)]
            if dgnn_test:
                df_dict["noise_t"] = df_dict["noise_t"] + [experiment.hp.noise_t for _ in range(amount_entries)]
            if experiment.noise != "uniform":
                df_dict["noise_ratio"] = df_dict["noise_ratio"] + experiment.noise_ratios[1:]
            else:
                df_dict["noise_ratio"] = df_dict["noise_ratio"] + experiment.noise_ratios
            df_dict["accuracy"] = df_dict["accuracy"] + accuracies
            df_dict["std"] = df_dict["std"] + stds
            INFO_LOGGER.info(f"Received Experiment Accuracies: {accuracies}")
            INFO_LOGGER.info("Added Experiment Result into DataFrame Dictionary.")
        INFO_LOGGER.info("Building Df of Experiments Results.")
        df = build_df_from_dict(df_dict)
        INFO_LOGGER.info("Parsing Df of Experiments Results into CSV.")
        parse_df_into_csv(df, os.path.join(PATH, f"output/results/experiments_results_{str(int(time.time()))}.csv"))

    def optimize_hps(self):
        for experiment in self.experiments:
            experiment.optimize_hp()
