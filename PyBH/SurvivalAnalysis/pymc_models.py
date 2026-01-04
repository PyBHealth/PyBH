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