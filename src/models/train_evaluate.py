"""fonctions d’entrainement et d’évaluation du modèle"""
import json
import os
import ssl
import urllib.request
import streamlit as st

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def json_formater(df, number):
    # Convertir le DataFrame en JSON
    json_df = df.iloc[(number - 1) : number, :].to_json(orient="split")
    # Formatons les données pour l'endpoints du modèle ML qui attend une structure spécifique du JSON
    json_to_py = json.loads(json_df)
    data_str = {"input_data": json_to_py}
    data = str.encode(json.dumps(data_str))
    return data


def allow_SelfSigned_Https(allowed):
    # bypass the server certificate verification on client side
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


def mlflow_model(body, url, api_key):
    allow_SelfSigned_Https(
        True
    )  # this line is needed if you use self-signed certificate in your scoring service.

    # Request data goes here
    # The example below assumes JSON formatting which may be updated
    # depending on the format your endpoint expects.
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

    # Replace this with the primary/secondary key or AMLToken for the endpoint
    if not api_key:
        raise ValueError("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {
        "Content-Type": "application/json",
        "Authorization": ("Bearer " + api_key),
        "azureml-model-deployment": "scoring1-1",
    }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", "ignore"))
    return result

def bytes_to_int(bytes_val):
    # Convertir bytes en string
    string_val = bytes_val.decode('utf-8')
    
    # On retire les caractères '[' et ']', puis on convertit le résultat en int
    output = string_val.strip('[]')
    #st.session_state.pred = output
    return output
