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
    def build_model(self, data, duration_col, event_col, coords=None, **kwargs):
        """
        Define the PyMC model structure (Priors and Likelihood).
        Must return a pm.Model() object.
        """
        pass

    def fit(self, data, duration_col, event_col, draws=2000, tune=1000, chains=4, coords=None, **kwargs):
        """
        Fit the model to the data using MCMC sampling.
        """
        self.duration_col = duration_col
        self.event_col = event_col
        self.last_data = data
        
        # 1. Initialize the PyMC model
        self.model = self.build_model(data, duration_col, event_col, coords=coords)
        
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

class Weibull(BayesianSurvivalModel):
    """
    Bayesian Weibull Survival Model implementation.
    Parameters: alpha (shape k), beta (scale eta).
    """

    def build_model(self, data, duration_col, event_col, coords=None, **kwargs):
        # Data split: Censored (0) vs Observed (1)
        observed = data[data[event_col] == 1][duration_col].values
        censored = data[data[event_col] == 0][duration_col].values
        
        # Prior for beta (scale) based on average survival time
        mean_time = data[duration_col].mean()

        with pm.Model(coords=coords) as model:
            # --- Priors ---
            # alpha (k): shape parameter. Controls if risk is increasing (>1) or decreasing (<1)
            alpha = pm.HalfNormal("alpha", sigma=2.0) 
            # beta (eta): scale parameter. Characteristic time of failure.
            beta = pm.HalfNormal("beta", sigma=mean_time * 5)
            
            # --- Likelihood ---
            # 1. Observed events: PDF f(t)
            if len(observed) > 0:
                pm.Weibull("obs_likelihood", alpha=alpha, beta=beta, observed=observed)
            
            # 2. Censored events: Survival function S(t)
            # Log(S(t)) = -(t/beta)^alpha
            if len(censored) > 0:
                log_surv_censored = - (censored / beta)**alpha
                pm.Potential("cens_likelihood", log_surv_censored)
                
        return model

    def predict_survival_function(self, times, credible_interval=0.95):
        """
        Predict S(t) = exp(-(t/beta)^alpha).
        Returns a DataFrame with mean survival and uncertainty bounds.
        """
        if self.idata is None:
            raise ValueError("Fit the model first.")
            
        # Extract posterior draws
        stacked = self.idata.posterior.stack(sample=("chain", "draw"))
        alpha_samples = stacked["alpha"].values
        beta_samples = stacked["beta"].values
        
        times = np.atleast_1d(times)
            
        # Compute survival curves: S(t) = exp(-(t/beta)^alpha)
        # Result shape: (num_samples, num_time_points)
        surv_curves = np.exp(- (times[np.newaxis, :] / beta_samples[:, np.newaxis]) ** alpha_samples[:, np.newaxis])
        
        # Statistics
        mean_surv = np.mean(surv_curves, axis=0)
        lower_bound = (1 - credible_interval) / 2
        upper_bound = 1 - lower_bound
        hdi = np.quantile(surv_curves, [lower_bound, upper_bound], axis=0)
        
        return pd.DataFrame({
            "time": times,
            "mean_survival": mean_surv,
            f"lower_{credible_interval}": hdi[0],
            f"upper_{credible_interval}": hdi[1]
        }).set_index("time")