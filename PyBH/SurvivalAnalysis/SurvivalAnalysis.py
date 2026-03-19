import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pymc_models import PyMCModel

class SurvivalAnalysis:
    """
    Workflow Manager pour l'analyse de survie.
    """

    def __init__(
        self, model: any, data: pd.DataFrame, time_col: str, event_col: str, **kwargs
    ):
        self.model = model
        self.idata = None

        self.is_bayesian = isinstance(model, PyMCModel)
        model_module = type(model).__module__
        self.is_lifelines = "lifelines" in model_module

        self.validate_inputs(data, time_col, event_col)
        df_clean = self._preprocess_data(data, time_col, event_col)

        self.durations = df_clean[time_col].values
        self.event_observed = df_clean[event_col].values
        self.X_df = df_clean.drop(columns=[time_col, event_col])

        # --- Model Dispatch Logic ---
        if self.is_bayesian:
            print("   -> Mode: Bayesian (PyMC)")
            coords = {
                "coeffs": self.X_df.columns.tolist(),
                "treated_units": df_clean.index.tolist(),
            }

            self.model.fit(
                self.X_df.values,
                self.durations,
                self.event_observed,
                coords=coords,
                **kwargs,
            )
            self.idata = self.model.idata

        elif self.is_lifelines:
            print("   -> Mode: Frequentist (Lifelines)")
            # Lifelines (ex: WeibullAFTFitter) nécessite un DataFrame consolidé
            df_lifelines = self.X_df.copy()
            df_lifelines[time_col] = self.durations
            df_lifelines[event_col] = self.event_observed
            
            self.model.fit(
                df_lifelines, duration_col=time_col, event_col=event_col, **kwargs
            )

        else:
            raise NotImplementedError("Unknown model type.")

    def validate_inputs(self, data: pd.DataFrame, time_col: str, event_col: str):
        if data.empty:
            raise ValueError("The input dataset is empty.")
        if time_col not in data.columns or event_col not in data.columns:
            raise ValueError(f"Columns '{time_col}' or '{event_col}' not found.")
        
        is_numeric = pd.api.types.is_numeric_dtype(data[time_col].dtype)
        is_datetime = pd.api.types.is_datetime64_any_dtype(data[time_col])

        if not is_numeric and not is_datetime:
            raise TypeError(f"Column '{time_col}' must be numeric or datetime format.")
        if data[event_col].sum() == 0:
            print("Warning: No events observed in the dataset. Convergence might fail.")

    def _preprocess_data(
        self, data: pd.DataFrame, time_col: str, event_col: str
    ) -> pd.DataFrame:
        """
        Nettoyage, encodage et standardisation stricte pour l'inférence Bayésienne.
        """
        df = data.copy()

        # Suppression totale des NaN pour éviter les échecs de calcul de gradient MCMC
        df = df.dropna()

        # Encodage One-Hot
        df = pd.get_dummies(df, drop_first=True)

        self.scalers = {}

        # Conversion des booléens
        cols_bool = df.select_dtypes(include=["bool"]).columns
        df[cols_bool] = df[cols_bool].astype(int)

        # Standardisation des variables continues (essentiel pour NUTS/Gibbs)
        for col in df.columns:
            if col not in [time_col, event_col] and not set(df[col].unique()).issubset({0, 1}):
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    self.scalers[col] = {'mean': mean, 'std': std}
                    df[col] = (df[col] - mean) / std

        return df


    def check_diagnostics(self) -> None:
        if self.idata is None:
            raise RuntimeError("Model has not been trained. Run .fit() first.")
        print("Checking convergence statistics...")

    def plot_survival_function(self, X_pred=None, ax=None, show_ci=True, **kwargs):
        """
        Trace la fonction de survie S(t) avec incertitudes structurelles.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # ==========================================================
        # 1. GESTION DE L'ENTRÉE ET STANDARDISATION (Le fix crucial)
        # ==========================================================
        if X_pred is None:
            # Si pas de profil fourni, on prend l'étudiant moyen
            X_eval = self.X_df.mean().to_frame().T
        else:
            # On s'assure que l'entrée est un DataFrame pour avoir le nom des colonnes
            if not isinstance(X_pred, pd.DataFrame):
                X_eval = pd.DataFrame(X_pred, columns=self.X_df.columns)
            else:
                X_eval = X_pred.copy()
                
            # Application de l'échelle mémorisée pendant l'entraînement !
            for col, scaler in getattr(self, 'scalers', {}).items():
                if col in X_eval.columns:
                    X_eval[col] = (X_eval[col] - scaler['mean']) / scaler['std']
        # ==========================================================

        label = kwargs.get("label", "Survie")
        color = kwargs.get("color", "blue")
        linestyle = kwargs.get("linestyle", "-")

        # ==========================================================
        # 2. TRACÉ MODE FRÉQUENTISTE (Lifelines)
        # ==========================================================
        if self.is_lifelines:
            surv_df = self.model.predict_survival_function(X_eval)
            times = surv_df.index
            ax.plot(times, surv_df.iloc[:, 0], label=label, color=color, linestyle=linestyle)

            if show_ci:
                mu = self.model.params_.values
                cov = self.model.variance_matrix_.values
                sim_params = np.random.multivariate_normal(mu, cov, 1000)
                
                cols = self.model.params_.index
                surv_sims = []
                
                for theta in sim_params:
                    param_dict = dict(zip(cols, theta))
                    log_lambda = param_dict.get(('lambda_', '_intercept'), 0)
                    for col in X_eval.columns:
                        if ('lambda_', col) in param_dict:
                            val = X_eval[col].values[0] 
                            log_lambda += param_dict[('lambda_', col)] * val
                    
                    lambda_ = np.exp(log_lambda)
                    rho_ = np.exp(param_dict.get(('rho_', '_intercept'), 0))
                    
                    surv_sims.append(np.exp(- (times / lambda_) ** rho_))
                
                surv_sims = np.array(surv_sims)
                ci_lower = np.quantile(surv_sims, 0.025, axis=0)
                ci_upper = np.quantile(surv_sims, 0.975, axis=0)
                
                ax.fill_between(times, ci_lower, ci_upper, color=color, alpha=0.15, label="CI Fréquentiste 95%")

        # ==========================================================
        # 3. TRACÉ MODE BAYÉSIEN (PyMC)
        # ==========================================================
        elif self.is_bayesian:
            if self.idata is None:
                raise RuntimeError("Modèle non entraîné.")

            t_max = self.durations.max()
            times = np.linspace(0, t_max, 100)
            
            # Prédiction sur l'entrée standardisée
            surv_df = self.model.predict_survival_function(times, X_eval)
            
            ax.plot(surv_df.index, surv_df["mean_survival"], label=label, color=color, linestyle=linestyle)
            
            if show_ci:
                ax.fill_between(surv_df.index, surv_df["lower_0.95"], surv_df["upper_0.95"], color=color, alpha=0.15, label="HDI Bayésien 95%")

        # ==========================================================
        # 4. FINITIONS DU GRAPHIQUE
        # ==========================================================
        ax.set_xlabel("Temps")
        ax.set_ylabel("Probabilité de survie $S(t)$")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    def summary(self):
        if self.is_bayesian:
            return self.model.print_summary()
        elif self.is_lifelines:
            return self.model.print_summary()