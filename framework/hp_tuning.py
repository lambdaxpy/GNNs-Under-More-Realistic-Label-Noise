import os
import sys
import torch
import random
import numpy as np

from framework.configparser import yamlparser, experimentfactory
from framework.experiment import experimentcollection
from framework.log.logger import get_info_logger, get_error_logger


INFO_LOGGER = get_info_logger(name=__name__)
ERROR_LOGGER = get_error_logger(name=__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_config_files(config_files):
    INFO_LOGGER.info("Parsing config files.")
    experiment_collection = experimentcollection.ExperimentCollection([])
    for config_file in config_files:
        config = yamlparser.parse_yaml_file_to_config_dict(config_file)
        experiment = experimentfactory.create_experiment(config)
        experiment_collection.add_experiment(experiment)
    INFO_LOGGER.info("Parsed config files.")
    return experiment_collection


def optimize_hps(experiment_collection: experimentcollection.ExperimentCollection):
    experiment_collection.optimize_hps()


def main(argv, argc):
    if argc < 2:
        ERROR_LOGGER.error("The number of program arguments is invalid.")
        raise ValueError("The number of program arguments is invalid.")
    config_path = argv[1]
    if os.path.isdir(config_path):
        config_files = [os.path.join(config_path, filename) for filename in os.listdir(config_path)]
    else:
        config_files = [config_path]

    INFO_LOGGER.info(f"Found {len(config_files)} configuration files.")

    experiment_collection = parse_config_files(config_files)
    set_seed(42)
    optimize_hps(experiment_collection)


if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
