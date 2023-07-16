""" fonctions regroupant les étapes de feature engineering """

import pandas as pd

options = [
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
]


def options_count(
    row: pd.Series,
) -> int:
    """count the number of the client options

    Args:
        row (pd.series): the DataFrame line

    Returns:
        int: the count of options
    """

    i = 0
    for col in options:
        if row[col] == "Yes":
            i += 1
        else:
            i += 0
    return i


def create_options_number(data: pd.DataFrame) -> pd.DataFrame:
    """create a new column with the number of the client options

    Args:
        data (pd.DataFrame): Dataset that should be modified

    Returns:
        pd.DataFrame: Initial dataset with the new column
    """
    data["options_number"] = data[options].apply(options_count, axis=1)
    return data


def columns_train_test(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> pd.DataFrame:
    """column naming replacement function to have a list of equivalent variable names between the two datasets

    Args:
        train_data (pd.DataFrame): original variables names
        test_data (pd.DataFrame): variable names that should be modified

    Returns:
        pd.DataFrame: Test dataset with equivalent variable names
    """
    train_cols = train_data.columns.tolist()
    test_cols = test_data.columns.tolist()

    for col1 in test_cols:
        for col2 in train_cols:
            if col2 in col1 and len(col2.split("_")) == len(col1.split("_")):
                test_data.rename(columns={col1: col2}, inplace=True)
            else:
                pass

    return test_data
