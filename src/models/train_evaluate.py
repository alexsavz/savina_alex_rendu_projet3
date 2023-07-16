"""fonctions d’entrainement et d’évaluation du modèle"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import requests

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


def data_features_selection(data: pd.DataFrame) -> pd.DataFrame:
    """Selection of the features for the model prediction

    Args:
        data (pd.DataFrame): DataFrame to filter

    Returns:
        data: DataFrame with filtered columns
    """
    features_list = [
        "gender",
        "age",
        "senior_citizen",
        "dependents",
        "number_of_dependents",
        "referred_a_friend",
        "number_of_referrals",
        "tenure_in_months",
        "offer",
        "avg_monthly_long_distance_charges",
        "multiple_lines",
        "internet_type",
        "avg_monthly_gb_download",
        "online_security",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_music",
        "unlimited_data",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_charge",
        "total_charges",
        "total_refunds",
        "total_extra_data_charges",
        "total_long_distance_charges",
        "total_revenue",
        "options_number",
    ]

    filtred_data = data[features_list]

    return filtred_data


# preprocessing Function
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """preprocessing the data set before the prediction

    Args:
        data (pd.DataFrame): DataFrame to preprocess

    Returns:
        data: DataFrame with normalized and encoded values
    """

    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns

    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
    )

    transformed_data = preprocessor.fit_transform(data)
    transformed_columns = [col.strip() for col in preprocessor.get_feature_names_out()]

    processed_data = pd.DataFrame(transformed_data, columns=transformed_columns)

    return processed_data


def load_model(path: str) -> dict:
    """Wrapper to import the model

    Args:
        path (str): File path

    Returns:
        object: model
    """
    url = path
    req = requests.get(url)
    req = req.text
    model = lgb.Booster(model_str=req)

    return model
