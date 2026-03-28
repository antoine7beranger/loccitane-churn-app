# NPS_impact/model/predict.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from NPS_impact.model.registry import load_pickle
from NPS_impact.params import COMMONE_FEATURES, DATASET_SPECIFIC_FEATURES

def load_artifacts():
    artifacts = {}

    # ÉTAPE CRUCIALE : On récupère le chemin absolu du dossier 'model'
    # os.path.dirname(__file__) donne le chemin vers NPS_impact/model/
    base_path = os.path.dirname(os.path.abspath(__file__))

    for nps in ["EC", "RE", "CS"]:
        # On construit le chemin complet vers chaque fichier
        prep_name = f"preprocessor_{nps}_churn.pkl"
        model_name = f"model_{nps}_churn.pkl"

        # On joint le dossier 'model' au nom du fichier
        prep_path = os.path.join(base_path, prep_name)
        model_path = os.path.join(base_path, model_name)

        artifacts[nps] = {
            "preprocessor": load_pickle(prep_path),
            "model":        load_pickle(model_path),
        }
    return artifacts

NPS_MAPPING = {
    "NPS_EC": "EC",
    "NPS_RE": "RE",
    "NPS_CS": "CS",
    "EC": "EC",
    "RE": "RE",
    "CS": "CS",
}


def predict(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    results = []

    # Normalise NPS_TYPE avant le filtre
    df["NPS_TYPE"] = df["NPS_TYPE"].map(NPS_MAPPING)

    for nps in ["EC", "RE", "CS"]:
        # 1. Filtre les lignes du bon NPS
        df_filtered = df[df["NPS_TYPE"] == nps].copy()
        if df_filtered.empty:
            continue

        # 2. Sélectionne les bonnes features
        features = COMMONE_FEATURES + DATASET_SPECIFIC_FEATURES[nps]
        X = df_filtered[features]

        # 3. Transform + Predict
        X_transformed = artifacts[nps]["preprocessor"].transform(X)
        df_filtered["CHURN_PRED"]  = artifacts[nps]["model"].predict(X_transformed)
        df_filtered["CHURN_PROBA"] = artifacts[nps]["model"].predict_proba(X_transformed)[:, 1]

        results.append(df_filtered)

    return pd.concat(results).sort_index()
