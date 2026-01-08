import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BayesianSurvivalModel(ABC):
    """
    Abstract Base Class for Bayesian Survival models using PyMC.
    Designed to mimic the Lifelines API.
    """
    def __init__(self):
        self.model = None
        self.idata = None  # Stores the InferenceData after fitting
        self.last_data = None
        self.duration_col = None
        self.event_col = None

    @abstractmethod
    def build_model(self, data, duration_col, event_col):
        """
        Define the PyMC model structure (Priors and Likelihood).
        Must return a pm.Model() object.
        """
        pass

    def fit(self, data, duration_col, event_col, draws=2000, tune=1000, chains=4, **kwargs):
        """
        Fit the model to the data using MCMC sampling.
        """
        self.duration_col = duration_col
        self.event_col = event_col
        self.last_data = data
        
        # 1. Initialize the PyMC model
        self.model = self.build_model(data, duration_col, event_col)
        
        # 2. Run the MCMC sampler
        with self.model:
            self.idata = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                **kwargs
            )
        return self
    
    @abstractmethod
    def predict_survival_function(self, times):
        """
        Calculate the survival probability S(t) for given time points.
        Returns a posterior distribution of survival curves.
        """
        pass

    def summary(self):
        """
        Print statistical summary of the posterior distributions.
        """
        if self.idata is None:
            raise ValueError("Model must be fitted before calling summary().")
        return az.summary(self.idata)

    def plot_traces(self):
        """
        Plot MCMC trace diagnostics.
        """
        if self.idata is None:
            raise ValueError("Model must be fitted before plotting.")
        az.plot_trace(self.idata)
        plt.tight_layout()
        plt.show()
    
    def score(self, data, duration_col, event_col):
        """
        Calculates the Concordance Index (C-index).
        Higher is better (0.5 is random, 1.0 is perfect).
        """
        pass

class BayesianPiecewiseCoxPH(BayesianSurvivalModel):
    """
    Bayesian Piecewise Constant Cox Proportional Hazards Model.
    Based on the formulation in Issue #8.
    """

    def __init__(self, time_intervals=5):
        super().__init__()
        self.cuts = time_intervals
        self.interval_bounds_ = None
        self._feature_names = None
        self._beta_means = None
        self._lambda_means = None

    def _data_expansion(self, data, duration_col, event_col):
        """
        Expands the dataset into 'long format'.
        """
        if isinstance(self.cuts, int):
            events = data[data[event_col] == 1]
            self.interval_bounds_ = np.unique(
                np.percentile(events[duration_col], np.linspace(0, 100, self.cuts + 1))
            )
            if self.interval_bounds_[0] > 0:
                self.interval_bounds_ = np.insert(self.interval_bounds_, 0, 0)
            self.interval_bounds_[-1] = max(self.interval_bounds_[-1], data[duration_col].max() + 1e-5)
        else:
            self.interval_bounds_ = np.array(sorted(self.cuts))
            if self.interval_bounds_[0] != 0:
                 self.interval_bounds_ = np.insert(self.interval_bounds_, 0, 0)

        long_data = []
        feature_cols = [c for c in data.columns if c not in [duration_col, event_col]]
                
        for _, row in data.iterrows():
            t_obs = row[duration_col]
            event = row[event_col]
                    
            for j in range(len(self.interval_bounds_) - 1):
                t_start = self.interval_bounds_[j]
                t_end = self.interval_bounds_[j+1]
                if t_obs <= t_start: break
                
                exposure = min(t_obs, t_end) - t_start
                interval_event = event if (t_obs <= t_end) else 0
                
                new_row = {col: row[col] for col in feature_cols}
                new_row['exposure'] = exposure
                new_row['interval_idx'] = j
                new_row['event_in_interval'] = interval_event
                long_data.append(new_row)
                
        return pd.DataFrame(long_data), feature_cols

    def build_model(self, data, duration_col, event_col):
        """
        Define the PyMC model structure.
        """
        # 1. Expand Data and explicitly get feature names
        df_long, feature_names = self._data_expansion(data, duration_col, event_col)
        self._feature_names = feature_names
        
        # 2. Extract arrays and force float types to avoid BLAS issues
        interval_indices = df_long['interval_idx'].values.astype(int)
        exposures = df_long['exposure'].values.astype(np.float64)
        events = df_long['event_in_interval'].values.astype(np.float64)
        X = df_long[feature_names].values.astype(np.float64)
        
        n_intervals = len(self.interval_bounds_) - 1
        
        coords = {
            "coeffs": feature_names,
            "intervals": np.arange(n_intervals),
            "obs_id": np.arange(len(df_long)) 
        }

        with pm.Model(coords=coords) as model:
            # Priors based on Issue #8 (Normal for beta, Gamma for lambda)
            beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")
            lambda_baseline = pm.Gamma("lambda0", alpha=0.01, beta=0.01, dims="intervals")
            
            # Map baseline hazard to intervals
            lambda_i = lambda_baseline[interval_indices]
            
            # Linear predictor
            log_hazard_ratio = pm.math.dot(X, beta)
            
            # Poisson rate: mu = lambda_0 * exp(X * beta) * exposure
            mu = pm.math.exp(pm.math.log(lambda_i) + log_hazard_ratio + pm.math.log(exposures))
            
            # Likelihood
            pm.Poisson("obs", mu=mu, observed=events, dims="obs_id")
            
        return model

    def fit(self, data, duration_col, event_col, draws=2000, tune=1000, chains=4, **kwargs):
        super().fit(data, duration_col, event_col, draws, tune, chains, **kwargs)
        self._beta_means = self.idata.posterior["beta"].mean(dim=["chain", "draw"]).values
        self._lambda_means = self.idata.posterior["lambda0"].mean(dim=["chain", "draw"]).values
        return self

    def predict_survival_function(self, X_pred, times=None):
        """
        Calculates S(t) = exp(-Cumulative_Hazard(t))
        """
        if self.idata is None:
            raise ValueError("Fit the model first.")
            
        if isinstance(X_pred, pd.DataFrame):
            X_pred = X_pred.values

        hr = np.exp(np.dot(X_pred, self._beta_means))
        
        if times is None:
            times = np.linspace(0, self.interval_bounds_[-1], 100)
        
        survival_curves = []
        for t in times:
            H0_t = 0
            for j in range(len(self.interval_bounds_) - 1):
                t_start = self.interval_bounds_[j]
                t_end = self.interval_bounds_[j+1]
                lam = self._lambda_means[j]
                
                if t <= t_start:
                    break
                elif t >= t_end:
                    H0_t += lam * (t_end - t_start)
                else:
                    H0_t += lam * (t - t_start)
            
            S_t = np.exp(-H0_t * hr)
            survival_curves.append(S_t)
            
        return pd.DataFrame(np.array(survival_curves), index=times, columns=range(len(X_pred)))