import os
import sys

import yaml
from pandas import DataFrame
import numpy as np
import dill

from src.logger import logging
from src.exception import CustomException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info("Exited the load_object method of utils")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save numpy array data to file.

    Args:
        file_path: The string location of file to be saved.
        array: The numpy array data to be saved.

    Returns:
        None.

    Raises:
        CustomException: If array is not successfully saved to the file location.

    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file.

    Args:
        file_path: The string location of file to be loaded.

    Returns:
        The loaded numpy array data.

    Raises:
        CustomException: If array is not successfully loaded from the file location.

    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
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
        raise CustomException(e, sys)
