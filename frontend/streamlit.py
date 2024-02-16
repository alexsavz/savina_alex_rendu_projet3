"""Fonctions pour l'affichage web du projet"""

# Frontend application using streamlit

import pandas as pd
import shap
import streamlit as st
# pip install streamlit-shap
from streamlit_shap import st_shap


def display_dashboard():
    """Affiche le contenu du dashboard."""
    st.set_page_config(layout="wide")

    # Titre du dashboard
    st.title("Churn prediction : identifier les clients fragiles")

    # Texte descriptif
    st.write("Voici un texte descriptif pour introduire le dashboard et son contenu.")

    # Utilisation de trois colonnes, où la colonne du milieu sert d'espace
    img1, spacer, img2 = st.columns([1, 0.2, 1])

    with img1:
        st.image("./frontend/assets/lift_curve.jpg", width=700)
        st.write("Description de la première image.")

    with img2:
        st.image("./frontend/assets/calibration_curve-min.jpg", width=550)
        st.write("Description de la deuxième image.")

    st.write("")
    scpace1, image, space2 = st.columns([0.2, 1, 0.2])
    with image:
        st.image("./frontend/assets/features_importance.jpg", width=900)
        st.write("Description de l'image large.")


# Représentation graphique summary_plot avec SHAP
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

    with st.expander("Barplot de la valeur moyenne en absolue du score SHAP pour chaque variable :"):
        st.image("./frontend/assets/plots_bar.jpg", width=1000)
    with st.expander("Beeswarm plot"):
        st_shap(shap.plots.beeswarm(shap_values), height=500, width=1000)

    return shap_values

def shap_tree_explainer(model, matrix, number, pred):
    number = number - 1
    # if 'pred' not in st.session_state:
    #     pred = 1
    # else:
    #     pred = st.session_state.pred
    # liste des colonnes
    model.params['objective'] = 'binary'
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(matrix)
    # Fonction pour appeler le graphique
    dependence_plot = shap.force_plot(tree_explainer.expected_value[int(pred)], shap_values[int(pred)][number], matrix.iloc[number,:])
    return dependence_plot

# def pred_output():
#     if 'pred' not in st.session_state:
#         st.session_state.pred = "sélectionnez un client pour réaliser une prédiction!" 
#     else:
#         st.session_state.pred

def select_input():
    st.session_state.number


def request_pred(df):
    st.header("Réaliser une requête API pour une prédiction :")
    if 'number' not in st.session_state:
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
            max_value=1887
        )
    
    with col2:
        st.dataframe(df.iloc[(number - 1) : number, :])

    return number

# def display_pred():
#     if 'pred' not in st.session_state:
#         st.session_state.pred = "sélectionnez un client pour réaliser une prédiction!" 
#     else:
#         st.session_state.pred
#     st.header(st.session_state.pred)

def pred_dashboard(number, shap_values, force_plot, pred):

    st.subheader("Output : " + pred, divider="rainbow")
    st.subheader("Nous pouvons à l'aide de la méthode SHAP afficher l'interprétation d'un client donné :")
    
    # Affichons les graphiques SHAP

    st_shap(shap.plots.waterfall(shap_values[number -1]), width=1000)
    st.write("Description de la première image.")

    st_shap(force_plot, width=1000) 
    st.write("Description de la deuxième image.")
