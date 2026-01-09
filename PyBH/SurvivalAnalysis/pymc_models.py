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
    """

    def __init__(self, time_intervals=5):
        super().__init__()
        self.cuts = time_intervals
        self.interval_bounds_ = None
        self._feature_names = None
        self._beta_means = None
        self._lambda_means = None

    def build_model(self, interval_indices, exposures, events, X):
        """
        Define the PyMC model structure using the pre-processed arrays.
        
        Parameters:
        - interval_indices: Array of interval IDs for each observation row
        - exposures: Time duration spent in the interval (Delta t)
        - events: Event indicator (1 if event occurred in this interval, 0 otherwise)
        - X: Covariate matrix (numpy array)
        """
        n_intervals = len(self.interval_bounds_) - 1
        
        # Coordinates for PyMC dimensions
        coords = {
            "coeffs": self._feature_names,
            "intervals": np.arange(n_intervals),
            "obs_id": np.arange(len(events)) 
        }

        with pm.Model(coords=coords) as model:
            # --- Priors ---
            # Regression coefficients (beta): Normal prior on beta 
            beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")
            
            # Baseline hazard (lambda_0): Gamma prior on lambda_j 
            # Corresponds to lambda_j ~ Gamma(alpha0, beta0) in the issue
            lambda_baseline = pm.Gamma("lambda0", alpha=0.01, beta=0.01, dims="intervals")
            
            # Map the baseline hazard to the specific intervals for each observation
            lambda_i = lambda_baseline[interval_indices]
            
            # --- Linear Predictor ---
            # lambda(t) = lambda_0(t) exp(x * beta)
            log_risk = pm.math.dot(X, beta) 
            
            # --- Log-Likelihood Construction  ---
            
            # Part A: Hazard contribution (only added if event occurred, delta=1)
            # Formula: delta_i * [ log(lambda_j) + beta * x_i ] 
            hazard_term = events * (pm.math.log(lambda_i) + log_risk)
            
            # Part B: Survival contribution (Cumulative Hazard)
            # Formula: - exp(beta * x_i) * lambda_j * Delta_t 
            # This accounts for the probability of surviving the duration of the interval
            survival_term = pm.math.exp(log_risk) * lambda_i * exposures
            
            # Combine terms: log L = Hazard_Term - Cumulative_Hazard_Term
            # We use pm.Potential to add this custom likelihood to the model 
            log_lik = hazard_term - survival_term
            
            pm.Potential("log_likelihood", log_lik)
            
        return model