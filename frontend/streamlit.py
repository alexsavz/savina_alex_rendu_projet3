"""Functions for the web display of the project"""

# Frontend application using streamlit

import pandas as pd
import shap
import streamlit as st

# pip install streamlit-shap
from streamlit_shap import st_shap


def display_dashboard():
    """Displays the content of the dashboard."""
    st.set_page_config(layout="wide")

    # Dashboard title
    st.title("Churn prediction : identifier les clients fragiles")

    # Description
    st.write(
        "Ce dashboard démonstratif est conçu pour permettre à l'utilisateur d'utiliser le modèle hébergé sur la plateforme cloud AzureML."
    )
    st.write(
        "Il s'agit d'une requête HTTPS à une URL qui sert de terminaison (Endpoint) pour générer une sortie (prédiction) affichée dans la partie Output."
    )
    st.write(
        "De plus, ce dashboard utilise la librairie SHAP (SHapley Additive exPlanations) pour fournir des visualisations qui expliquent les contributions de chaque variables à la prédiction individuelle, offrant ainsi une meilleure compréhension des décisions du modèle."
    )


# Graphical representation summary_plot with SHAP
def shap_summary_plot(model: object, masker: pd.DataFrame) -> pd.DataFrame:
    """Explain the dataset with graphical visualization

    Args:
        model (object): trained model
        masker (pd.DataFrame): data matrix

    Returns:
        explainer object: Build a new explainer for the passed model.
    """
    explainer = shap.Explainer(model, masker)
    shap_values = explainer(masker, check_additivity=False)

    with st.expander(
        "Barplot de la valeur moyenne en absolue du score SHAP pour chaque variable :"
    ):
        st.image("./frontend/assets/plots_bar.jpg", width=1000)
    with st.expander("Beeswarm plot"):
        st_shap(shap.plots.beeswarm(shap_values), height=500, width=1000)

    return shap_values


def shap_tree_explainer(model, matrix, number, pred):
    number = number - 1
    # liste of columns
    model.params["objective"] = "binary"
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(matrix)
    # Fonction pour appeler le graphique
    dependence_plot = shap.force_plot(
        tree_explainer.expected_value[int(pred)],
        shap_values[int(pred)][number],
        matrix.iloc[number, :],
    )
    return dependence_plot


def select_input():
    st.session_state.number


def request_pred(df):
    st.header("Réaliser une requête API pour une prédiction :")
    if "number" not in st.session_state:
        number = 1
    else:
        number = st.session_state.number

    col1, col2 = st.columns([1, 1])
    with col1:
        st.number_input(
            "Sélectionner un client :",
            key="number",
            on_change=select_input,
            placeholder="Sélection du client",
            step=1,
            min_value=1,
            max_value=1887,
        )

    with col2:
        st.dataframe(df.iloc[(number - 1) : number, :])

    return number


def pred_dashboard(number, shap_values, force_plot, pred):
    st.subheader("Output : " + str(pred), divider="rainbow")
    st.subheader(
        "Nous pouvons à l'aide de la méthode SHAP afficher l'interprétation d'un client donné :"
    )
    st.markdown(
        """#### ***Un score proche de :red[1] indique que le client est à :red[risque de résiliation] et :blue[inversement pour 0].***"""
    )

    st_shap(shap.plots.waterfall(shap_values[number - 1]), width=1000)
    st.write(
        "Le graphique SHAP waterfall est un type de visualisation qui montre comment chaque caractéristique du modèle contribue à déplacer la prédiction à partir de la valeur de base (la prédiction moyenne pour le dataset de train) jusqu'à la prédiction. Les variables qui influencent la prédiction à augmenter vers 1 sont affichées en rouge, celles qui poussent la prédiction à diminuer vers 0 sont en bleu."
    )

    st_shap(force_plot, width=1000)
    st.write(
        "Le force plot est une visualisation qui affiche la même information sous une forme différente, mettant en évidence la contribution de chaque variable à la prédiction individuelle par l'accumulation des flèches."
    )
