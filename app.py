import os
import sys
import streamlit as st
import pandas as pd
from NPS_impact.model.predict import load_artifacts, predict

# Configuration de la page
st.set_page_config(page_title="Churn Prevention", layout="wide")

# --- STYLE L'OCCITANE ---
st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    .stButton>button {
        background-color: #FBB900;
        color: #1D1D1B;
        border-radius: 5px;
        font-weight: bold;
    }
    h1 {
        color: #1D1D1B;
        border-bottom: 3px solid #FBB900;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE DE RECOMMANDATIONS ---
def donner_recommandation(row):
    if row['CHURN_PRED'] == 1:
        if row['CHURN_PROBA'] > 0.8:
            return "🔴 Alerte Critique : Contact immédiat"
        return "🟠 Risque modéré : Envoyer offre promo"
    return "🟢 Client fidèle"

# --- INTERFACE ---
st.title("Tableau de Bord Prévention du Churn")

# Chargement des artefacts (modèles)
artifacts = load_artifacts()

# 1. Chargement du fichier client
st.subheader("📁 Étape 1 : Importer les données clients")
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :", data.head(3))

    # 2. Bouton de prédiction
    if st.button("Lancer l'analyse de Churn"):
        with st.spinner('Analyse en cours...'):
            try:
                # Appel de la fonction de prédiction (depuis ton fichier predict.py)
                # On suppose que predict renvoie le DataFrame avec les colonnes CHURN_PRED et CHURN_PROBA
                results = predict(data, artifacts)

                # Application de la logique de recommandation
                results['RECOMMANDATION'] = results.apply(donner_recommandation, axis=1)

                # --- AFFICHAGE DES RÉSULTATS ---
                st.success("Analyse terminée !")

                # Métriques clés
                col1, col2, col3 = st.columns(3)
                nb_churners = results[results['CHURN_PRED'] == 1].shape[0]
                col1.metric("Clients à risque", nb_churners)
                col2.metric("Taux de Churn estimé", f"{(nb_churners/len(results)*100):.1f}%")

                # Table des résultats
                st.subheader("📋 Liste des clients et actions recommandées")

                # On colore le tableau pour une meilleure lisibilité
                def color_churn(val):
                    color = '#ff4b4b' if val == 1 else '#28a745'
                    return f'color: {color}; font-weight: bold'

                # On définit le dictionnaire de formatage
# {1:.0%} -> multiplie par 100 et ajoute le signe % avec 0 décimale
# {1:.0f} -> affiche le nombre avec 0 chiffre après la virgule
                formatter = {
                             'CHURN_PROBA': '{:.0%}',
                             'SC_Status_NPS': '{:.0f}'    # Assure-toi que ta colonne s'appelle bien NPS_SCORE
                }

# Affichage du tableau stylisé
                st.dataframe(
                results[['VIP_ID', 'NPS_TYPE', 'SC_Status_NPS', 'CHURN_PROBA', 'CHURN_PRED', 'RECOMMANDATION']]
                .style
                .format(formatter)           # Applique le % et l'arrondi
                .map(color_churn, subset=['CHURN_PRED']) # Garde tes couleurs
                )

                # Option d'export
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les prédictions (CSV)",
                    data=csv,
                    file_name='predictions_churn.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

else:
    st.info("Veuillez importer un fichier CSV contenant les données clients pour commencer.")

# --- FOOTER ---
st.markdown("---")
st.caption("Outil interne - Direction Marketing & CRM")
