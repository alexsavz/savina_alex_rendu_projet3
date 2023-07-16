"""fonctions d’import de données"""

import yaml
import pandas as pd


def import_yaml_config(location: str) -> dict:
    """Wrapper to easily import YAML

    Args:
        location (str): File path

    Returns:
        dict: YAML content as dict
    """
    with open(location, "r", encoding="utf-8") as stream:
        dict_config = yaml.safe_load(stream)

    return dict_config


def import_data(path: str) -> pd.DataFrame:
    """Import churn datasets

    Args:
        path (str): File location

    Returns:
        pd.DataFrame: churn dataset
    """

    data = pd.read_csv(path)

    return data


def merge_data(data1: pd.DataFrame, data2: pd.DataFrame, column: str) -> pd.DataFrame:
    """merge dataset

    Args:
        pd.DataFrame: churn dataset

    Returns:
        pd.DataFrame: datasets merged
    """
    data = data1.merge(data2, how="inner", on=column)

    return data
