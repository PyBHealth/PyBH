import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import de tes classes
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../..')))

# 1. On importe TES classes
from PyBH.SurvivalAnalysis.SurvivalAnalysis import SurvivalAnalysis
from PyBH.SurvivalAnalysis.pymc_models import WeibullPH # ou Cox, au choix !
# ==========================================
# 1. DESIGN ET FOND D'ÉCRAN
# ==========================================
st.set_page_config(page_title="Simulateur de Survie", page_icon="🎓", layout="centered")

# CSS pour mettre une image de fond (tu peux mettre une URL ou un chemin local)
fond_ecran = """
<style>
.stApp {
    background-image: url("https://unsplash.com/fr/photos/medecin-tenant-un-stethoscope-rouge-hIgeoQjS_iE");
    background-size: cover;
}
/* Un petit fond semi-transparent pour que le texte reste lisible */
.main .block-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
}
</style>
"""
st.markdown(fond_ecran, unsafe_allow_html=True)

st.title("🎓 Simulateur de Survie Étudiante")
st.markdown("Découvre tes chances de survivre au semestre sans faire de burn-out !")

# ==========================================
# 2. LOGIQUE DU MODÈLE (Cachée et optimisée)
# ==========================================

@st.cache_resource
def charger_modele():
    # Ton générateur
    np.random.seed(42)
    sommeil = np.random.normal(6.5, 1.2, 1000).clip(3, 10)
    cafes = np.random.poisson(3, 1000).clip(0, 10)
    soirees = np.random.poisson(1.5, 1000).clip(0, 5)
    sport = np.random.poisson(1, 1000).clip(0, 7)
    bde = np.random.binomial(1, 0.3, 1000)

    risque = (-0.6*(sommeil-6.5) + 0.15*cafes + 0.4*soirees - 0.4*sport + 0.3*bde)
    taux = np.exp(risque) * 0.03
    
    temps = -np.log(np.random.uniform(0, 1, 1000)) / taux
    df = pd.DataFrame({
        'sommeil': np.round(sommeil, 1), 'cafes': cafes, 'soirees': soirees,
        'sport': sport, 'bde': bde,
        'time': np.maximum(0.1, np.round(np.minimum(temps, 24), 1)),
        'event': (temps <= 24).astype(int)
    })
    
    # Entraînement
    modele = WeibullPH()
    sa = SurvivalAnalysis(model=modele, data=df, time_col='time', event_col='event', progressbar=False)
    return sa

# On charge le modèle en fond
with st.spinner("L'IA révise ses cours... (Chargement initial)"):
    sa_app = charger_modele()

# ==========================================
# 3. L'INTERFACE UTILISATEUR (Interactive)
# ==========================================
st.subheader("Rentre ton profil :")

col1, col2 = st.columns(2)

with col1:
    sommeil_in = st.slider("🛌 Heures de sommeil", min_value=3.0, max_value=10.0, value=7.0, step=0.5)
    cafes_in = st.number_input("☕ Nombre de cafés/jour", min_value=0, max_value=15, value=2)
    bde_in = st.selectbox("🎉 Membre du BDE / Asso ?", options=["Non", "Oui"])
    bde_val = 1 if bde_in == "Oui" else 0

with col2:
    soirees_in = st.slider("🍻 Soirées par semaine", min_value=0, max_value=7, value=1)
    sport_in = st.slider("🏃‍♂️ Séances de sport/sem", min_value=0, max_value=7, value=1)

# Le bouton pour lancer la prédiction
if st.button("🔮 Calculer mon espérance de vie", use_container_width=True):
    
    profil = np.array([[sommeil_in, cafes_in, soirees_in, sport_in, bde_val]])
    
    st.markdown("---")
    st.subheader("📊 Tes Résultats")
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Courbe moyenne (par défaut)
    sa_app.plot_survival_function(ax=ax, label="Étudiant Moyen", color="gray", linestyle="--")
    # Courbe du visiteur
    sa_app.plot_survival_function(X_pred=profil, ax=ax, label="TA SURVIE", color="#FF4B4B", linewidth=3)
    
    ax.set_title("Probabilité de survie au cours du semestre")
    ax.set_xlim(0, 24)
    ax.axvline(x=12, color='orange', linestyle=':', label='Partiels')
    ax.legend()
    
    # Affichage du plot dans Streamlit
    st.pyplot(fig)