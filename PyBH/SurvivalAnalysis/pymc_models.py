import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class PyMCModel(ABC):
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
    def build_model(self, data, duration_col, event_col, coords):
        """
        Define the PyMC model structure (Priors and Likelihood).
        Must return a pm.Model() object.
        """
        pass

    def fit(self, data, duration_col, event_col, coords, draws=2000, tune=1000, chains=4, **kwargs):
        """
        Fit the model to the data using MCMC sampling.
        """
        self.duration_col = duration_col
        self.event_col = event_col
        self.last_data = data
        
        # 1. Initialize the PyMC model
        self.model = self.build_model(data, duration_col, event_col, coords)
        
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

class Cox(PyMCModel):
    r"""
    Define the PyMC model structure using a Piece-wise Exponential Model (PEM).
        
    This implementation exploits the mathematical equivalence between the Cox 
    Proportional Hazards model and a Poisson regression. 

    **Mathematical Equivalence:**
    
    The hazard rate for individual :math:`i` in time interval :math:`j` is:
    
    .. math::
        \lambda_{ij} = \lambda_j \exp(X_i \beta)
        
    where :math:`\lambda_j` is the baseline hazard for interval :math:`j`. 
    In a survival model, the log-likelihood contribution of an observation 
    is given by:
    
    .. math::
        \log L_{ij} = d_{ij} \log(\lambda_{ij}) - \int_{t \in I_j} \lambda_{ij} dt
        
    Under the assumption that :math:`\lambda_j` is constant over the interval 
    duration :math:`\Delta t_{ij}`, the integral simplifies to:
    
    .. math::
        \log L_{ij} = d_{ij} (\log(\Delta t_{ij}) + \log(\lambda_j) + X_i \beta) - (\Delta t_{ij} \lambda_j e^{X_i \beta})
        
    This is identical (up to a constant :math:`\log(\Delta t_{ij})`) to the 
    log-likelihood of a Poisson distribution :math:`\text{Poisson}(\mu_{ij})` 
    where:
    
    .. math::
        \mu_{ij} = \Delta t_{ij} \cdot \lambda_j \cdot \exp(X_i \beta)
    """

    def __init__(self, interval_length=5, priors=None):
        super().__init__()
        # 1. Define the piece-wise intervals
        self.interval_length = interval_length 
        
        # 2. Allow custom priors for flexibility 
        self.priors = priors if priors else {
            "beta_sigma": 10.0,       # Prior std dev for regression coeffs
            "lambda_alpha": 0.01,     # Gamma alpha for baseline hazard
            "lambda_beta": 0.01       # Gamma beta for baseline hazard
        }
        
        self.interval_bounds_ = None
        self._feature_names = None


    def build_model(self, interval_indices, exposures, events, X, coords):
        r"""
        Parameters
        ----------
        interval_indices : array-like
            Integer indices mapping each observation to its respective time interval.
        exposures : array-like
            The duration :math:`\Delta t_{ij}` spent by the individual in the interval 
            (Time at Risk).
        events : array-like
            Binary indicator :math:`d_{ij}` (1 if the event occurred, 0 otherwise).
        X : ndarray
            Matrix of covariates (features).
        coords : dict
            PyMC coordinates for dimension naming (e.g., {"coeffs": ..., "intervals": ...}).

        Returns
        -------
        model : pm.Model
            The compiled PyMC model object.
        """
        n_intervals = len(self.interval_bounds_) - 1
        
        with pm.Model(coords=coords) as model:
            # --- Priors ---
            # Regression coefficients (beta): Normal prior
            beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")
            
            # Baseline hazard (lambda_0): Gamma prior 
            # We use independent priors for each interval
            lambda0 = pm.Gamma("lambda0", 
                                alpha=self.priors["lambda_alpha"], 
                                beta=self.priors["lambda_beta"], 
                                dims="intervals")
            
            # Map the baseline hazard to the specific intervals for each observation
            lambda_i = lambda0[interval_indices]
            
            # --- Poisson Means Calculation ---
            # The mean of the Poisson process for a specific interval is:
            # mu = exposure * lambda_0(t) * exp(x * beta)
            
            # Calculate the risk score: exp(X * beta)
            risk = pm.math.exp(pm.math.dot(X, beta))
            
            # Calculate mu
            mu = exposures * lambda_i * risk
            
            # --- Likelihood ---
            # We use the Poisson approximation for the piece-wise exponential model
            # observed=events matches the 'd_ij' (death indicator) from the PyMC example
            pm.Poisson("likelihood", mu=mu, observed=events)
            
        return model
    
    def predict_survival_function(self, times):
        """
        Calculate the survival probability S(t) for given time points.
        Returns a posterior distribution of survival curves.
        """
        pass
