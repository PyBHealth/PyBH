from abc import ABC, abstractmethod

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


class PyMCModel(ABC):
    """
    Abstract Base Class for Bayesian Survival models using PyMC.
    Designed to mimic the Lifelines API.
    """

    def __init__(self):
        self.model = None
        self.idata = None  # Stores the InferenceData after fitting
        self.duration_col = None
        self.event_col = None
        self._feature_names = None

    @abstractmethod
    def build_model(self, data, duration_col, event_col, coords=None, **kwargs):
        """
        Define the PyMC model structure (Priors and Likelihood).
        Must return a pm.Model() object.
        """
        pass

    def fit(
        self,
        data,
        duration_col,
        event_col,
        coords,
        draws=2000,
        tune=1000,
        chains=4,
        **kwargs,
    ):
        """
        Fit the model to the data using MCMC sampling.
        """
        self.duration_col = duration_col
        self.event_col = event_col

        # 1. Initialize the PyMC model
        self.model = self.build_model(data, duration_col, event_col, coords=coords)

        # 2. Run the MCMC sampler
        with self.model:
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, **kwargs)
        return self

    @abstractmethod
    def predict_survival_function(self, times, X_new):
        """
        Calculate the survival probability S(t) for given time points.
        Returns a posterior distribution of survival curves.
        """
        pass

    def print_summary(self):
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
        print("Method not yet implemented.")


class Cox(PyMCModel):
    r"""
    This class defines the Bayesian Cox Proportional Hazard model using the
    Poisson equivalence (Piecewise Exponential Model).

    It models the survival process as a set of Poisson distributions where the
    expected number of events :math:`\mu_{ij}` for patient i in interval j is:

    .. math::
        \mu_{ij} = \Delta t_{ij} \cdot \lambda_j \cdot \exp(X_i \beta)

    where:

    - :math:`\Delta t_{ij}` is the time (exposure) patient i spent in interval j.
    - :math:`\lambda_j` is the baseline hazard for the interval j.
    - :math:`X_i` is the vector of covariates for patient i.
    - :math:`\beta` is the vector of coefficients (log-hazard ratios) associated
    with the covariates.

    Parameters
    ----------
    cutpoints : list or np.array
        Ordered timepoints defining the intervals for the piecewise constant
        baseline hazard.

    Examples
    --------

    >>> import pymc
    >>> import pandas
    >>> from PyBH.SurvivalAnalysis.SurvivalAnalysis import SurvivalAnalysis
    >>> from PyBH.SurvivalAnalysis.pymc_models import Cox

    >>> # Typical dataset for survival analysis
    >>> data = pandas.read_csv(pymc.get_data("mastectomy.csv"))

        # Define intervals: 0-10, 10-20, 20+
        model = Cox(cutpoints=[10, 20])

        # Launch analysis
        analysis = SurvivalAnalysis(model=model,
                                    data=data,
                                    time_col="time",
                                    event_col="event",)

        # Plot obtained survival function
        analysis.plot_survival_function()
    """

    def __init__(self, cutpoints, priors=None):
        super().__init__()
        self.cutpoints = np.sort(np.unique(np.concatenate(([0], cutpoints))))
        self.interval_bounds_ = np.concatenate((self.cutpoints, [np.inf]))

        self.priors = {"beta_sigma": 1.0, "lambda_alpha": 0.01, "lambda_beta": 0.01}

        if priors:
            self.priors.update(priors)

        self._feature_names = None

    def _transform_to_long_format(self, X, times, events):
        """
        Converts survival data to long format for piecewise constant hazard modeling.
        Each subject is expanded into multiple rows, one for each time interval they
        entered, tracking their exposure time and whether the event occurred.
        """
        n_samples = len(X)
        n_intervals = len(self.interval_bounds_) - 1

        long_idx, long_exp, long_evt, long_X = [], [], [], []

        for i in range(n_samples):
            # t_obs : Time of the event for i
            # e_obs : 0 if censored, 1 if event
            t_obs, e_obs = times[i], events[i]

            for j in range(n_intervals):
                # Extract j-th interval's delimitation
                t_start, t_end = self.interval_bounds_[j], self.interval_bounds_[j + 1]

                # If event occurred before beginning of time interval, Break
                if t_obs <= t_start:
                    break

                # Time spent at risk within the interval
                exposure = min(t_obs, t_end) - t_start
                is_event = 1.0 if (t_obs <= t_end and e_obs == 1) else 0.0

                long_idx.append(j)
                long_exp.append(exposure)
                long_evt.append(is_event)
                long_X.append(X[i])

        return (
            np.array(long_idx, dtype=int),
            np.array(long_exp, dtype=float),
            np.array(long_evt, dtype=float),
            np.array(long_X, dtype=float),
        )

    def build_model(self, interval_indices, exposures, events, X_long, coords):
        """
        Constructs the Bayesian Piecewise Exponential Model using PyMC.
        """
        with pm.Model(coords=coords) as model:
            # Priors for the regression coefficients (log-hazard ratios)
            beta = pm.Normal(
                "beta", mu=0, sigma=self.priors["beta_sigma"], dims="coeffs"
            )

            # Baseline hazard for each discrete time interval
            lambda0 = pm.Gamma(
                "lambda0",
                alpha=self.priors["lambda_alpha"],
                beta=self.priors["lambda_beta"],
                dims="intervals",
            )

            # Compute log-risk for each observation
            log_risk = (X_long * beta[None, :]).sum(axis=-1)

            # Expected value for the Poisson likelihood:
            mu = exposures * lambda0[interval_indices] * pm.math.exp(log_risk)
            pm.Poisson("obs", mu=mu, observed=events)

        return model

    def fit(
        self, X, time, event, coords=None, draws=2000, tune=1000, chains=2, **kwargs
    ):
        """
        Fits the Bayesian Piecewise Exponential Model to the provided survival data.
        """
        # Define feature names for the model coordinates
        self._feature_names = coords.get(
            "coeffs",
            [f"v{i}" for i in range(X.shape[1] if hasattr(X, "shape") else len(X[0]))],
        )

        # Convert from Wide (1 row/subject) to Long (N rows/subject)
        # This is required to model the survival process as a Poisson counting process
        idx, exp, evt, X_long = self._transform_to_long_format(X, time, event)

        # Define Model Dimensions
        model_coords = {
            "coeffs": self._feature_names,
            "intervals": [f"Int_{i}" for i in range(len(self.interval_bounds_) - 1)],
        }
        self.model = self.build_model(idx, exp, evt, X_long, model_coords)

        with self.model:
            self.idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=1, **kwargs
            )

        return self

    def predict_survival_function(self, times, X_new):
        """
        Predicts the survival function for new samples at given time points.
        Calculates S(t) = exp(-H(t)), where H(t) is the cumulative hazard.
        """
        if self.idata is None:
            raise ValueError("Model not fitted.")

        # Extract posterior samples for baseline hazards and coefficients
        post = self.idata.posterior
        lambdas = post["lambda0"].stack(sample=("chain", "draw")).values.T
        betas = post["beta"].stack(sample=("chain", "draw")).values.T

        X_arr = X_new.values if hasattr(X_new, "values") else X_new

        # Calculate the relative risk scores for each posterior sample
        risk_scores = np.exp(np.dot(betas, X_arr.T))

        # Compute cumulative baseline hazard by integrating the piecewise
        # constant hazard
        cum_h0 = np.zeros((betas.shape[0], len(times)))
        for t_idx, t in enumerate(times):
            for j in range(len(self.interval_bounds_) - 1):
                t_start, t_end = self.interval_bounds_[j], self.interval_bounds_[j + 1]

                # If the target time 't' is beyond the start of this interval
                if t > t_start:
                    # Add hazard contribution: (rate * time_spent_in_interval)
                    cum_h0[:, t_idx] += lambdas[:, j] * (min(t, t_end) - t_start)

        # Final survival probability
        return np.exp(-risk_scores[:, :, np.newaxis] * cum_h0[:, np.newaxis, :])


class Weibull(PyMCModel):
    """
    Bayesian Weibull Survival Model implementation.
    Parameters: alpha (shape k), beta (scale eta).
    """

    def build_model(self, data, duration_col, event_col, coords=None, **kwargs):
        # Data split: Censored (0) vs Observed (1)
        observed = duration_col[event_col == 1]
        censored = duration_col[event_col == 0]

        # Prior for beta (scale) based on average survival time
        mean_time = duration_col.mean()

        with pm.Model(coords=coords) as model:
            # --- Priors ---
            # alpha (k): shape parameter. Controls if risk is increasing (>1)
            # or decreasing (<1)
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
                log_surv_censored = -((censored / beta) ** alpha)
                pm.Potential("cens_likelihood", log_surv_censored)

        return model

    def predict_survival_function(self, times, X_new, credible_interval=0.95):
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
        surv_curves = np.exp(
            -(
                (times[np.newaxis, :] / beta_samples[:, np.newaxis])
                ** alpha_samples[:, np.newaxis]
            )
        )

        # Statistics
        mean_surv = np.mean(surv_curves, axis=0)
        lower_bound = (1 - credible_interval) / 2
        upper_bound = 1 - lower_bound
        hdi = np.quantile(surv_curves, [lower_bound, upper_bound], axis=0)

        return pd.DataFrame(
            {
                "time": times,
                "mean_survival": mean_surv,
                f"lower_{credible_interval}": hdi[0],
                f"upper_{credible_interval}": hdi[1],
            }
        ).set_index("time")
