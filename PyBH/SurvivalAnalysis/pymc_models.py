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
    - :math:`\beta` is the vector of coefficients (log-hazard ratios) associated \
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


class WeibullPH(PyMCModel):
    r"""
    Weibull Proportional Hazards (PH) Model.

    Models the instantaneous hazard rate as:
    .. math::
        h(t|x) = h_0(t) \exp(x \beta)

    Where the baseline hazard $h_0(t)$ follows a Weibull distribution.
    
    In PyMC's Weibull parameterization (alpha, beta=scale), the PH assumption 
    implies that the scale parameter varies per individual:
    .. math::
        \text{scale}(x) = \text{scale}_0 \times \exp\left(-\frac{x \beta}{\alpha}\right)
    """

    def fit(self, X, duration_col, event_col, coords=None, draws=2000, tune=1000, chains=2, **kwargs):
        """
        Fits the Bayesian model using MCMC sampling.
        
        Args:
            X (array-like): Matrix of covariates (standardized).
            duration_col (array-like): Time to event or censorship.
            event_col (array-like): Event indicator (1=Observed, 0=Censored).
            coords (dict): Dimension names for ArviZ/Xarray.
        """
        # 1. Format inputs to NumPy arrays
        X_arr = np.asarray(X)
        time_arr = np.asarray(duration_col)
        event_arr = np.asarray(event_col)

        # Force X to be 2D (N, P) even if there is only one feature
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if X_arr.shape[0] != len(time_arr):
            raise ValueError(f"Dimension mismatch: X has {X_arr.shape[0]} rows, Time has {len(time_arr)}.")

        # 2. Handle Coordinates for coefficients naming
        if coords is None:
            coords = {}
        
        if "coeffs" not in coords:
            n_features = X_arr.shape[1]
            coords["coeffs"] = [f"v{i}" for i in range(n_features)]

        # Track observation IDs for downstream diagnostics
        coords["obs_id"] = np.where(event_arr == 1)[0]

        # 3. Build and Sample
        self.model = self.build_model(X_arr, time_arr, event_arr, coords=coords)
        
        with self.model:
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, **kwargs)
            
        return self

    def build_model(self, X, time, event, coords=None, **kwargs):
        """
        Defines the PyMC probabilistic graph.
        """
        # Split indices for observed vs censored data
        obs_idx = np.where(event == 1)[0]
        cens_idx = np.where(event == 0)[0]
        
        # Heuristic for scale prior to aid convergence
        mean_time = np.mean(time)

        with pm.Model(coords=coords) as model:
            # --- Priors ---
            
            # Alpha (shape): k
            # k > 1: Hazard increases over time
            # k < 1: Hazard decreases over time
            alpha = pm.HalfNormal("alpha", sigma=2.0)
            
            # Lambda0 (baseline scale): sigma_0
            # Represents the scale for an "average" individual if X is centered
            lambda0 = pm.HalfNormal("lambda0", sigma=mean_time * 2)

            # Betas (log-hazard ratios)
            # Normal(0,1) is a standard weakly informative prior for scaled data
            betas = pm.Normal("beta", mu=0, sigma=1.0, dims="coeffs")

            # --- Weibull PH Parameterization ---
            
            # 1. Linear Predictor: eta = X * beta
            linear_predictor = pm.math.dot(X, betas)
            
            # 2. Map PH to AFT scale
            # scale(x) = lambda0 * exp( - (X * beta) / alpha )
            scale = lambda0 * pm.math.exp(-linear_predictor / alpha)

            # --- Likelihood ---
            
            # A. Observed Events -> Probability Density Function (PDF)
            if len(obs_idx) > 0:
                pm.Weibull(
                    "obs",
                    alpha=alpha,
                    beta=scale[obs_idx],
                    observed=time[obs_idx],
                    dims="obs_id"
                )

            # B. Censored Events -> Survival Function (CCDF)
            # Log S(t) = - (t / scale)^alpha
            if len(cens_idx) > 0:
                log_surv_censored = -((time[cens_idx] / scale[cens_idx]) ** alpha)
                pm.Potential("cens_likelihood", log_surv_censored)

        return model

    def predict_survival_function(self, times, X_new, credible_interval=0.95):
        """
        Calculates predicted survival curves S(t|x) for new data.
        """
        if self.idata is None:
            raise ValueError("Model must be fitted before predicting.")

        X_arr = np.asarray(X_new)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
            
        times = np.atleast_1d(times)

        # Extract posterior samples
        post = self.idata.posterior
        alpha_s = post["alpha"].stack(sample=("chain", "draw")).values 
        lambda0_s = post["lambda0"].stack(sample=("chain", "draw")).values 
        beta_s = post["beta"].stack(sample=("chain", "draw")).values 

        # 1. Linear Predictor
        lp = np.dot(X_arr, beta_s)

        # 2. Adjusted scale
        scale_s = lambda0_s * np.exp(-lp / alpha_s)

        # 3. Survival Curves (Broadcasting)
        t_br = times[np.newaxis, :, np.newaxis]
        sc_br = scale_s[:, np.newaxis, :]
        al_br = alpha_s[np.newaxis, np.newaxis, :]
        
        surv_raw = np.exp(- (t_br / sc_br) ** al_br)
        
        # On extrait uniquement le patient cible (index 0)
        surv_patient = surv_raw[0] 
        
        # On calcule la moyenne
        mean_surv = np.mean(surv_patient, axis=1)

        # On calcule les intervalles de confiance
        lower_bound = (1 - credible_interval) / 2
        upper_bound = 1 - lower_bound
        hdi = np.quantile(surv_patient, [lower_bound, upper_bound], axis=1)

        # On renvoie le bon dictionnaire avec les colonnes exactes
        return pd.DataFrame({
            "mean_survival": mean_surv,
            f"lower_{credible_interval}": hdi[0],
            f"upper_{credible_interval}": hdi[1]
        }, index=times)

    def score(self, X, duration_col, event_col):
        """
        Calculates the Concordance Index (C-index).
        
        Returns:
            float: 0.5 (random) to 1.0 (perfect).
        """
        try:
            from lifelines.utils import concordance_index
        except ImportError:
            raise ImportError("Package 'lifelines' is required for scoring.")

        if self.idata is None:
            raise ValueError("Model must be fitted.")

        X_arr = np.asarray(X)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(-1, 1)

        # 1. Get mean posterior coefficients
        beta_mean = self.idata.posterior["beta"].mean(dim=["chain", "draw"]).values
        
        # 2. Calculate Risk Score
        risk_scores = np.dot(X_arr, beta_mean)
        
        # 3. Calculate C-index
        # Since High Risk = Low Survival, we use -risk_scores for the concordance index.
        return concordance_index(duration_col, -risk_scores, event_col)