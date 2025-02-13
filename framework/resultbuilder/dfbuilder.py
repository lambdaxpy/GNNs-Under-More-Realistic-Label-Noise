import pandas as pd
from framework.log.logger import get_info_logger


INFO_LOGGER = get_info_logger(name=__name__)


def build_df_from_dict(df_dict: dict):
    df = pd.DataFrame.from_dict(df_dict)
    INFO_LOGGER.info(f"Built df from dict: {df_dict}")
    return df

