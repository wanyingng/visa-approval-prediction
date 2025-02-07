import os
import sys

import yaml
from pandas import DataFrame

from src.logger import logging
from src.exception import CustomException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool=False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)


def drop_columns(df: DataFrame, cols: list)-> DataFrame:
    """
    Drop the columns of a pandas DataFrame.

    Args:
        df: The pandas DataFrame.
        cols: The list of columns to be dropped.

    Returns:
        The resulting pandas DataFrame without the columns.

    Raises:
        CustomException: If cols is not dropped.

    """
    logging.info("Dropping columns from DataFrame")
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info("Columns successfully dropped from DataFrame")
        return df
    except Exception as e:
        raise CustomException(e, sys) from e
