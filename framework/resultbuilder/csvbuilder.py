import pandas as pd
from framework.log.logger import get_info_logger


INFO_LOGGER = get_info_logger(name=__name__)


def parse_df_into_csv(df: pd.DataFrame, csv_path: str) -> None:
    df.to_csv(csv_path)
    INFO_LOGGER.info(f"Converted df to csv: {csv_path}")

