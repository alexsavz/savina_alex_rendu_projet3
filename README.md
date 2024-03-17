<h1 align="center">Projet de scoring : Identifier les clients fragiles</h1>

<p align="center">
  <a href="https://github.com/alexsavz/savina_alex_rendu_projet3/actions/workflows/ci.yaml">
    <img src="https://github.com/alexsavz/savina_alex_rendu_projet3/actions/workflows/ci.yaml/badge.svg" alt="Build, test and push"/>
  </a>
</p>

<p align="center"><i>Modèle de machine learning : lightGBM</i></p>

<h2 align="center">Dataset</h2>

<p align="center"><i>Telco costumer churn</i></p>

<h2 align="center">Problématique</h2>

<p align="center">
<b>Contexte :</b>
<br>- Domaine de la télécommunication, mission avec l'équipe marketing
<br>- Identifier les clients fragiles et décrire leur profil
<br>- Extraire les variables les plus pertinentes à partir des données clients
</p>

<p align="center">
<b>Mission :</b>
<br>- Réaliser une analyse exploratoire
<br>- Réaliser une analyse prédictive
<br>- Concevoir un projet portable et reproductible
<br>- Réaliser un dashboard intéractif
</p>

<h2 align="center">Dashboard</h2>
<table align="center">
  <tr>
    <td align="center" valign="top">
      Dashboard intéractif <br/>
      <a href="https://alexsavina-scoring-clientfragile.streamlit.app/">https://alexsavina-scoring-clientfragile.streamlit.app/</a> <br/><br>
      <a href="https://alexsavina-scoring-clientfragile.streamlit.app/">
        <img alt="Todo App" src="/frontend/assets/dashboard-min.png" width="200px" style="max-width:100%; border-radius: 10px;"/>
      </a>
    </td>
  </tr>
</table>

## Tech stack

**Analyse des données:** pandas, numpy, scipy
**Représentation graphique:** matplotlib, seaborn, plotly
**Modelisation:** scikit-learn, lightgbm, shap
**Mise en production:** Docker, Github Actions, mlflow, Streamlit

## Notebook

1. Exploration du jeu de données
   - Description des données
   - Contrôle de la qualité des données
   - Analyse exploratoire
   - Présélection de variables
2. Création et optimisation du modèle
   - Préprocessing des données d'évaluation
   - Pipeline de transformation
   - Optimisation du modèle
3. Evaluation et explicativité des modèles
   - Courbe de lift
   - Score de spiegelhalter
   - Courbe de calibration
   - Importance des variables
4. Sérialisation du meilleur modèle
   - Sauvegarde locale
   - Log du modèle avec MLflow
5. Prédiction sur l'échantillon de test

## Réutilisation

Commandes pour utiliser le projet:

```python
pip install -r requirements.txt
python main.py
```

La prédiction sera enregistrée sous un format csv dans le dossier data.
