from yaml import safe_load, dump
from ml_collections import config_dict
from framework.log import logger


ERROR_LOGGER = logger.get_error_logger(name=__name__)


def get_yaml_content(file_path: str) -> dict:
    if not file_path.endswith('.yaml'):
        ERROR_LOGGER.error("The config file(s) is/are not YAML file(s).")
        raise ValueError("The config file(s) is/are not YAML file(s).")

    with open(file_path, "r") as yaml_file:
        content = safe_load(yaml_file)
    return content


def convert_yaml_to_config_dict(yaml: dict) -> config_dict.ConfigDict:
    return config_dict.ConfigDict(yaml)


def parse_yaml_file_to_config_dict(file_path: str) -> config_dict.ConfigDict:
    return convert_yaml_to_config_dict(get_yaml_content(file_path))


def parse_hp_dict_to_yaml(hp_dict: dict, file_path: str):
    with open(file_path, "w") as yaml_file:
        yaml_output = dump(hp_dict)
        yaml_file.write(yaml_output)

