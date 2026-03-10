import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

# Import de tes classes
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../..')))

# 1. On importe TES classes
from PyBH.SurvivalAnalysis.SurvivalAnalysis import SurvivalAnalysis
from PyBH.SurvivalAnalysis.pymc_models import WeibullPH 

# ==========================================
# 1. DESIGN ET FOND D'ÉCRAN
# ==========================================
st.set_page_config(page_title="Simulateur de Carrière", page_icon="💼", layout="centered")

# Changement de l'image de fond pour une ambiance "Bureaux / Entreprise"
fond_ecran = """
<style>
.stApp {
    background-image: url("");
    background-size: cover;
}
.main .block-container {
    background-color: rgba(255, 255, 255, 0.90);
    padding: 2rem;
    border-radius: 15px;
}
</style>
"""
st.markdown(fond_ecran, unsafe_allow_html=True)

st.title("💼 Simulateur de Carrière & Démission")
st.markdown("Estimation de la durée moyenne en poste avant une démission (modèle basé sur les données RH IBM réelles).")

# ==========================================
# 2. LOGIQUE DU MODÈLE (Kagglehub intégré)
# ==========================================
@st.cache_resource
def charger_modele():
    # 1. Téléchargement automatique via Kaggle
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
    
    # Trouver le fichier CSV dans le dossier téléchargé
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    df_brut = pd.read_csv(os.path.join(path, csv_file))
    
    # 2. Sélection des colonnes pertinentes pour le stand
    colonnes = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'OverTime', 'WorkLifeBalance', 'YearsAtCompany', 'Attrition']
    df = df_brut[colonnes].copy()
    
    # --- ADAPTATION FRANCE ---
    # On applique un coefficient de 0.65 pour ramener les salaires US à une échelle Française réaliste
    df['MonthlyIncome'] = df['MonthlyIncome'] * 0.65

    # 3. Nettoyage et encodage pour les maths
    # Attrition : Yes = 1 (A démissionné), No = 0 (Est resté)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    # OverTime : Yes = 1, No = 0
    df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Éviter le temps 0 exact pour le modèle de Weibull
    df['YearsAtCompany'] = np.maximum(0.1, df['YearsAtCompany'])
    
    # 4. Entraînement
    modele = WeibullPH()
    sa = SurvivalAnalysis(model=modele, data=df, time_col='YearsAtCompany', event_col='Attrition', progressbar=False)
    
    return sa

# On charge le modèle en fond avec un message sympa
with st.spinner("L'IA analyse les dossiers RH d'IBM... (Chargement en cours)"):
    sa_app = charger_modele()

# ==========================================
# 3. L'INTERFACE UTILISATEUR (Interactive)
# ==========================================
st.subheader("Configuration du profil :")

col1, col2 = st.columns(2)

with col1:
    age_in = st.slider("🧑‍🎓 Âge de l'employé", min_value=18, max_value=60, value=25)
    distance_in = st.slider("🚗 Distance Domicile-Travail (km)", min_value=1, max_value=50, value=10)
    
    # --- CURSEUR ADAPTÉ À LA FRANCE ---
    income_in = st.number_input("💰 Salaire Mensuel Brut (€)", min_value=1500, max_value=10000, value=2000, step=100)

with col2:
    overtime_in = st.selectbox("⏱️ Heures Supplémentaires effectuées ?", options=["Non", "Oui"])
    overtime_val = 1 if overtime_in == "Oui" else 0
    
    wlb_in = st.slider("⚖️ Équilibre Vie Pro/Perso", min_value=1, max_value=4, value=3, help="1 = Mauvais, 4 = Excellent")

# Le bouton pour lancer la prédiction
if st.button("🔮 Lancer l'estimation de survie en entreprise", use_container_width=True):
    
    # L'ordre DOIT correspondre aux colonnes gardées plus haut (sans YearsAtCompany ni Attrition)
    # Ordre : Age, DistanceFromHome, MonthlyIncome, OverTime, WorkLifeBalance
    profil = np.array([[age_in, distance_in, income_in, overtime_val, wlb_in]])
    
    st.markdown("---")
    st.subheader("📊 Résultats de la prédiction")
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Courbe moyenne (par défaut)
    sa_app.plot_survival_function(ax=ax, label="Moyenne IBM", color="gray", linestyle="--")
    
    # Courbe du visiteur
    sa_app.plot_survival_function(X_pred=profil, ax=ax, label="PROFIL PERSONNALISÉ", color="#0066cc", linewidth=3)
    
    ax.set_title("Probabilité de maintien en poste au fil des années", fontsize=14, fontweight='bold')
    ax.set_xlabel("Temps passé dans l'entreprise (Années)")
    ax.set_ylabel("Probabilité de ne pas démissionner")
    ax.set_xlim(0, 30) # On limite à 30 ans de carrière dans la même boîte
    
    ax.legend()
    
    # Affichage du plot dans Streamlit
    st.pyplot(fig)