import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
# --- FIX CRITIQUE POUR PYMC ---
# Empêche les conflits entre le multiprocessing de PyMC et le multithreading de NumPy (BLAS)
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# ==========================================
# 1. DÉFINITION DES CLASSES (FRAMEWORK)
# ==========================================

class PyMCModel(ABC):
    """Classe Abstraite pour uniformiser l'interface."""
    def __init__(self):
        self.model = None
        self.idata = None 
        self.last_data = None
        self.duration_col = None
        self.event_col = None
        self.covariates = None

    @abstractmethod
    def build_model(self, data, duration_col, event_col, coords=None, **kwargs):
        pass

    def fit(self, data, duration_col, event_col, covariates=None, draws=1000, tune=1000, chains=2, coords=None, **kwargs):
        self.duration_col = duration_col
        self.event_col = event_col
        self.covariates = covariates
        self.last_data = data
        
        # Gestion automatique des coords pour les graphiques
        if covariates:
            if coords is None:
                coords = {"coeffs": covariates}
            else:
                coords["coeffs"] = covariates

        print(f"--- Construction du modèle avec covariates: {covariates} ---")
        self.model = self.build_model(data, duration_col, event_col, covariates=covariates, coords=coords)
        
        print("--- Sampling MCMC ---")
        with self.model:
            # target_accept augmenté pour éviter les divergences sur des petits datasets
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9, **kwargs)
        return self

    @abstractmethod
    def predict_survival_function(self, times, X_pred=None):
        pass

    def summary(self):
        if self.idata is None: raise ValueError("Fit model first")
        return az.summary(self.idata)

class WeibullAFT(PyMCModel):
    """
    Modèle AFT : ln(T) = beta0 + beta*X + sigma*W
    Interprétation : beta > 0 allonge la survie.
    """
    def build_model(self, data, duration_col, event_col, covariates=None, coords=None, **kwargs):
        t = data[duration_col].values
        e = data[event_col].values
        X = data[covariates].values if covariates else np.zeros((len(data), 1))
        
        # Séparation observé/censuré
        obs_idx = np.where(e == 1)[0]
        cens_idx = np.where(e == 0)[0]

        with pm.Model(coords=coords) as model:
            # Priors
            alpha = pm.HalfNormal("alpha", sigma=2.0) # Paramètre de forme
            beta0 = pm.Normal("beta0", mu=np.log(t.mean()), sigma=2.0)
            
            if covariates:
                betas = pm.Normal("betas", mu=0.0, sigma=1.0, dims="coeffs")
                lin_pred = beta0 + pm.math.dot(X, betas)
            else:
                lin_pred = beta0

            # AFT : Scale = exp(mu)
            scale = pm.math.exp(lin_pred)

            # Likelihood
            if len(obs_idx) > 0:
                pm.Weibull("obs", alpha=alpha, beta=scale[obs_idx], observed=t[obs_idx])
            if len(cens_idx) > 0:
                log_surv = - (t[cens_idx] / scale[cens_idx])**alpha
                pm.Potential("cens", log_surv)
        return model

    def predict_survival_function(self, times, X_pred=None, credible_interval=0.95):
        if self.covariates:
            X_val = np.array(X_pred) if X_pred is not None else self.last_data[self.covariates].mean().values
        else:
            X_val = np.array([0])

        post = self.idata.posterior
        alpha = post["alpha"].stack(sample=("chain", "draw")).values
        beta0 = post["beta0"].stack(sample=("chain", "draw")).values
        
        if self.covariates:
            betas = post["betas"].stack(sample=("chain", "draw")).values
            lin_pred = beta0 + np.dot(X_val, betas)
        else:
            lin_pred = beta0
            
        scale = np.exp(lin_pred)
        times = np.atleast_1d(times)
        # S(t) = exp(-(t/scale)^alpha)
        surv_curves = np.exp(- (times[:, np.newaxis] / scale[np.newaxis, :]) ** alpha[np.newaxis, :])
        
        return pd.DataFrame({
            "time": times,
            "mean_survival": surv_curves.mean(axis=1)
        }).set_index("time")

class WeibullPH(PyMCModel):
    """
    Modèle PH : h(t) = h0(t) * exp(beta*X)
    Interprétation : beta > 0 augmente le risque (réduit la survie).
    """
    def build_model(self, data, duration_col, event_col, covariates=None, coords=None, **kwargs):
        t = data[duration_col].values
        e = data[event_col].values
        X = data[covariates].values if covariates else np.zeros((len(data), 1))
        obs_idx = np.where(e == 1)[0]
        cens_idx = np.where(e == 0)[0]

        with pm.Model(coords=coords) as model:
            alpha = pm.HalfNormal("alpha", sigma=2.0)
            lambda_base = pm.HalfNormal("lambda_base", sigma=t.mean()*2) # Scale de base

            if covariates:
                betas_ph = pm.Normal("betas", mu=0.0, sigma=1.0, dims="coeffs")
                risk_score = pm.math.dot(X, betas_ph)
            else:
                risk_score = 0.0

            # Transformation clé PH -> Weibull Scale
            # Scale = lambda_base * exp(- risk / alpha)
            scale = lambda_base * pm.math.exp(- risk_score / alpha)

            if len(obs_idx) > 0:
                pm.Weibull("obs", alpha=alpha, beta=scale[obs_idx], observed=t[obs_idx])
            if len(cens_idx) > 0:
                log_surv = - (t[cens_idx] / scale[cens_idx])**alpha
                pm.Potential("cens", log_surv)
        return model

    def predict_survival_function(self, times, X_pred=None, credible_interval=0.95):
        if self.covariates:
            X_val = np.array(X_pred) if X_pred is not None else self.last_data[self.covariates].mean().values
        else:
            X_val = np.array([0])

        post = self.idata.posterior
        alpha = post["alpha"].stack(sample=("chain", "draw")).values
        lambda_base = post["lambda_base"].stack(sample=("chain", "draw")).values
        
        if self.covariates:
            betas = post["betas"].stack(sample=("chain", "draw")).values
            risk_score = np.dot(X_val, betas)
        else:
            risk_score = 0.0
            
        scale = lambda_base * np.exp(- risk_score / alpha)
        times = np.atleast_1d(times)
        surv_curves = np.exp(- (times[:, np.newaxis] / scale[np.newaxis, :]) ** alpha[np.newaxis, :])
        
        return pd.DataFrame({
            "time": times,
            "mean_survival": surv_curves.mean(axis=1)
        }).set_index("time")

# ==========================================
# 2. SCRIPT PRINCIPAL
# ==========================================
# ==========================================
# 2. SCRIPT PRINCIPAL (CORRIGÉ)
# ==========================================

if __name__ == "__main__":
    print("1. Chargement des données Mastectomy...")
    try:
        # Essai chargement local/cache PyMC
        df = pd.read_csv(pm.get_data("mastectomy.csv"))
    except:
        # Fallback si le cache est vide ou l'API inaccessible
        url = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/mastectomy.csv"
        df = pd.read_csv(url)

    print("Colonnes trouvées :", df.columns.tolist())

    # --- CORRECTION ROBUSTE ---
    # On cherche la colonne (peu importe son nom exact) et on la convertit
    if 'metastasized' in df.columns:
        source_col = 'metastasized'
    elif 'metastasis' in df.columns:
        source_col = 'metastasis'
    else:
        # Fallback ultime (3e colonne)
        source_col = df.columns[2]

    print(f"Traitement de la colonne source : '{source_col}'")
    print(f"Valeurs uniques avant nettoyage : {df[source_col].unique()}")

    # Conversion intelligente "yes"/"no" -> 1/0
    # Si c'est déjà numérique, ça restera numérique. Si c'est "yes", ça deviendra 1.
    df['metastized'] = (df[source_col].astype(str) == 'yes').astype(int)
    
    # Sécurité : Si la colonne était déjà 0/1 mais en int, le test == 'yes' a donné 0 partout.
    # On vérifie si on a tout cassé (somme = 0)
    if df['metastized'].sum() == 0 and df[source_col].dtype != object:
         print("La colonne était déjà numérique, conversion directe...")
         df['metastized'] = df[source_col].astype(int)

    print(f"Valeurs uniques après nettoyage : {df['metastized'].unique()}")
    print(df.head())

    # Configuration
    duration_col = "time"
    event_col = "event" 
    covariates = ["metastized"] 

    # ------------------------------------------
    # 2. Entraînement Modèle AFT (Accelerated Failure Time)
    # ------------------------------------------
    print("\n>>> Entraînement WEIBULL AFT (Temps)...")
    # ... (Le reste est identique)
    model_aft = WeibullAFT()
    model_aft.fit(df, duration_col, event_col, covariates=covariates, chains=2, draws=1000)
    
    summ_aft = model_aft.summary()
    print("\n--- Résumé AFT ---")
    print(summ_aft.loc[['betas[metastized]', 'alpha']])

    # ------------------------------------------
    # 3. Entraînement Modèle PH (Proportional Hazards)
    # ------------------------------------------
    print("\n>>> Entraînement WEIBULL PH (Risque)...")
    model_ph = WeibullPH()
    model_ph.fit(df, duration_col, event_col, covariates=covariates, chains=2, draws=1000)
    
    summ_ph = model_ph.summary()
    print("\n--- Résumé PH ---")
    print(summ_ph.loc[['betas[metastized]', 'alpha']])

    # ------------------------------------------
    # 4. Vérification Mathématique
    # ------------------------------------------
    print("\n>>> Vérification de la relation mathématique (PH vs AFT)")
    
    beta_aft_mean = summ_aft.loc['betas[metastized]', 'mean']
    alpha_mean = summ_aft.loc['alpha', 'mean']
    beta_ph_mean = summ_ph.loc['betas[metastized]', 'mean']
    
    predicted_ph_beta = - alpha_mean * beta_aft_mean
    
    print(f"Beta AFT réel : {beta_aft_mean:.3f}")
    print(f"Alpha (Shape) : {alpha_mean:.3f}")
    print(f"Beta PH prédit (-alpha * beta_aft) : {predicted_ph_beta:.3f}")
    print(f"Beta PH réel (obtenu par le modèle) : {beta_ph_mean:.3f}")
    
    # ------------------------------------------
    # 5. Visualisation
    # ------------------------------------------
    print("\n>>> Génération des graphiques...")
    t_plot = np.linspace(0, df[duration_col].max(), 100)
    
    surv_aft = model_aft.predict_survival_function(t_plot, X_pred=[1])
    surv_ph = model_ph.predict_survival_function(t_plot, X_pred=[1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(surv_aft.index, surv_aft["mean_survival"], label="Weibull AFT (Metastized=1)", color="blue", linestyle="--")
    plt.plot(surv_ph.index, surv_ph["mean_survival"], label="Weibull PH (Metastized=1)", color="red", alpha=0.7)
    
    plt.title("Comparaison des courbes de survie prédites (Patient avec Métastases)")
    plt.xlabel("Temps (Mois)")
    plt.ylabel("Probabilité de Survie S(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()
    print("Terminé.")