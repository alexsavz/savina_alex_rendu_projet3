"""fonctions d’export de données"""

import pandas as pd


def save_data(data1, data2) -> pd.DataFrame:
    """Export churn dataset

    Args:
        data1 (pd.DataFrame): dataset to transform,
        data2 (pd.DataFrame): dataset to transform

    Returns:
        pd.DataFrame: churn dataset
    """

    data2 = pd.DataFrame(data2)
    col = data2.columns.tolist()
    data2.rename({col[0]: "churn_value"}, axis=1, inplace=True)
    y_pred_test = pd.concat([data1, data2], axis=1, verify_integrity=True)
    csv = y_pred_test.to_csv(
        "./dataset/eval_df_telco_customer_churn_services_pred.csv", index=False
    )

    return csv
