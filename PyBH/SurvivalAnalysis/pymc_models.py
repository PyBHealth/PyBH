import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BayesianSurvivalModel(ABC):
    def __init__(self):
        self.model = None
        self.idata = None 
        self._feature_names = None

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, X, y, coords=None, **kwargs):
        pass
    
    @abstractmethod
    def predict_survival_function(self, X_new, times):
        pass

    # --- ADDED DIAGNOSTIC METHODS ---
    def plot_traces(self, **kwargs):
        """Visualizes the MCMC chains using ArviZ."""
        if self.idata is None:
            raise ValueError("Model must be fitted before plotting traces.")
        return az.plot_trace(self.idata, **kwargs)

    def summary(self, **kwargs):
        """Returns a summary table of the posterior distribution."""
        if self.idata is None:
            raise ValueError("Model must be fitted before generating a summary.")
        return az.summary(self.idata, **kwargs)

class Cox(BayesianSurvivalModel):
    def __init__(self, cutpoints, priors=None):
        super().__init__()
        self.cutpoints = np.sort(np.unique(np.concatenate(([0], cutpoints))))
        self.interval_bounds_ = np.concatenate((self.cutpoints, [np.inf]))
        
        self.priors = {
            "beta_sigma": 1.0,
            "lambda_alpha": 0.01,
            "lambda_beta": 0.01
        }
        
        if priors:
            self.priors.update(priors)
            
        self._feature_names = None

    def _transform_to_long_format(self, X, times, events):
        n_samples = X.shape[0]
        n_intervals = len(self.interval_bounds_) - 1
        
        long_idx, long_exp, long_evt, long_X = [], [], [], []
        
        for i in range(n_samples):
            t_obs, e_obs = times[i], events[i]
            for j in range(n_intervals):
                t_start, t_end = self.interval_bounds_[j], self.interval_bounds_[j+1]
                if t_obs <= t_start: break
                
                exposure = min(t_obs, t_end) - t_start
                is_event = 1.0 if (t_obs <= t_end and e_obs == 1) else 0.0
                
                long_idx.append(j)
                long_exp.append(exposure)
                long_evt.append(is_event)
                long_X.append(X[i])

        return (np.array(long_idx, dtype=int), 
                np.array(long_exp, dtype=float), 
                np.array(long_evt, dtype=float), 
                np.array(long_X, dtype=float))

    def fit(self, X, y, coords=None, draws=2000, tune=1000, chains=2, **kwargs):
        times, events = y[:, 0], y[:, 1]
        self._feature_names = coords.get("coeffs", [f"v{i}" for i in range(X.shape[1])])
        
        idx, exp, evt, X_long = self._transform_to_long_format(X, times, events)
        
        model_coords = {
            "coeffs": self._feature_names,
            "intervals": [f"Int_{i}" for i in range(len(self.interval_bounds_) - 1)]
        }
        self.model = self.build_model(idx, exp, evt, X_long, model_coords)
        
        with self.model:
            # Note: cores=1 is often required on Windows to avoid BrokenPipeErrors
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, **kwargs)
        return self

    def build_model(self, interval_indices, exposures, events, X_long, coords):
        with pm.Model(coords=coords) as model:
            beta = pm.Normal("beta", mu=0, sigma=self.priors["beta_sigma"], dims="coeffs")
            lambda0 = pm.Gamma("lambda0", alpha=self.priors["lambda_alpha"], 
                               beta=self.priors["lambda_beta"], dims="intervals")
            
            log_risk = (X_long * beta[None, :]).sum(axis=-1)
            mu = exposures * lambda0[interval_indices] * pm.math.exp(log_risk)
            
            pm.Poisson("obs", mu=mu, observed=events)
        return model

    def predict_survival_function(self, X_new, times):
        if self.idata is None: raise ValueError("Model not fitted.")
        post = self.idata.posterior
        lambdas = post["lambda0"].stack(sample=("chain", "draw")).values.T 
        betas = post["beta"].stack(sample=("chain", "draw")).values.T      
        
        X_arr = X_new.values if hasattr(X_new, "values") else X_new
        risk_scores = np.exp(np.dot(betas, X_arr.T)) 
        
        cum_h0 = np.zeros((betas.shape[0], len(times)))
        for t_idx, t in enumerate(times):
            for j in range(len(self.interval_bounds_) - 1):
                t_start, t_end = self.interval_bounds_[j], self.interval_bounds_[j+1]
                if t > t_start:
                    cum_h0[:, t_idx] += lambdas[:, j] * (min(t, t_end) - t_start)

        return np.exp(-risk_scores[:, :, np.newaxis] * cum_h0[:, np.newaxis, :])