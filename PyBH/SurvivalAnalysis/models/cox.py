import numpy as np
import pymc as pm
import pytensor.tensor as pt
from .pymc_model import PyMCModel  

class CoxModel(PyMCModel):
"""
Bayesian Cox proportional hazards model with:
• piecewise constant baseline hazard
• Gamma priors for λ_j
• Normal priors for β
• Custom log-likelihood
Uses the standard PyMCModel .fit() / .predict() / .score() API.
"""

def __init__(
    self,
    n_intervals=10,
    beta_prior_sd=5.0,
    lambda_prior_shape=0.1,
    lambda_prior_rate=0.1,
    **kwargs,
):
    """
    Parameters
    ----------
    n_intervals : int
        Number of piecewise-constant hazard segments.
    beta_prior_sd : float
        Stddev of normal prior on β.
    lambda_prior_shape : float
        Shape of the Gamma prior on λ_j.
    lambda_prior_rate : float
        Rate of the Gamma prior on λ_j.
    """
    super().__init__(**kwargs)

    self.n_intervals = n_intervals
    self.beta_prior_sd = beta_prior_sd
    self.lambda_prior_shape = lambda_prior_shape
    self.lambda_prior_rate = lambda_prior_rate

    # Filled after fit
    self.interval_edges_ = None
    self.model_ = None

# ----------------------------------------------------------------------
# Utility: compute interval index for each time
# ----------------------------------------------------------------------
def _compute_intervals(self, time):
    """Given event/censoring times, return interval boundaries and indices."""
    # Cut times into quantile-based bins
    edges = np.quantile(time, np.linspace(0, 1, self.n_intervals + 1))
    # Ensure unique edges
    edges = np.unique(edges)

    # Digitize times into intervals
    idx = np.digitize(time, edges[1:], right=True)

    return edges, idx

# ----------------------------------------------------------------------
# Main PyMC model builder
# ----------------------------------------------------------------------
def build_model(self, X, time, event):
    """
    Build and return a PyMC model for the Cox PH likelihood.
    """
    X = np.asarray(X)
    time = np.asarray(time)
    event = np.asarray(event).astype(int)

    # Prepare piecewise-constant intervals
    edges, interval_idx = self._compute_intervals(time)
    self.interval_edges_ = edges

    # Precompute exposure per interval:
    # For each subject i and interval j:
    #   exposure[i,j] = length of time_i spent inside interval j.
    n = len(time)
    J = len(edges) - 1

    exposure = np.zeros((n, J))
    for j in range(J):
        left = edges[j]
        right = edges[j + 1]
        for i in range(n):
            t = time[i]
            exposure[i, j] = max(0.0, min(t, right) - left)

    with pm.Model() as model:
        # Priors on baseline hazard λ_j
        lam = pm.Gamma(
            "lambda",
            alpha=self.lambda_prior_shape,
            beta=self.lambda_prior_rate,
            shape=J,
        )

        # Priors on β
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=self.beta_prior_sd,
            shape=X.shape[1],
        )

        # Linear predictor
        eta = pt.dot(X, beta)

        # Log-likelihood for each subject:
        #   log p_i = 1(event_i) * (log λ_{I_i} + η_i)
        #             - sum_j λ_j * exposure[i,j] * exp(η_i)
        baseline_term = pt.sum(lam * exposure, axis=1)  # sum_j λ_j * exposure
        mu = baseline_term * pt.exp(eta)

        loglik = (
            event * (pt.log(lam[interval_idx]) + eta)
            - mu
        )

        pm.Potential("cox_loglike", loglik)

    self.model_ = model
    return model

# ----------------------------------------------------------------------
# Prediction: expected hazard, survival, etc.
# ----------------------------------------------------------------------
def predict(self, X_new, return_type="hazard"):
    """
    Predict hazard or risk score from posterior samples.

    Parameters
    ----------
    return_type : {"hazard", "risk"}
    """
    if self.trace_ is None:
        raise RuntimeError("Model must be fitted before prediction.")

    X_new = np.asarray(X_new)
    beta_samples = self.trace_.posterior["beta"].stack(draws=("chain", "draw")).values

    # Risk score exp(Xβ)
    risk = np.exp(X_new @ beta_samples)

    if return_type == "risk":
        return risk.mean(axis=1)

    elif return_type == "hazard":
        lam_samples = self.trace_.posterior["lambda"].stack(draws=("chain", "draw")).values
        baseline = lam_samples.sum(axis=0)  # total baseline hazard per sample
        hazard = risk * baseline  # approximate hazard
        return hazard.mean(axis=1)

    else:
        raise ValueError("return_type must be 'hazard' or 'risk'.")

# ----------------------------------------------------------------------
# Score: partial log-likelihood approximation
# ----------------------------------------------------------------------
def score(self, X, time, event):
    """
    Score = expected log-likelihood under the posterior.
    """
    if self.trace_ is None:
        raise RuntimeError("Model must be fitted before scoring.")

    # Recompute model log-likelihood
    with self.model_:
        return pm.logp(self.model_, self.trace_.posterior.stack(draws=("chain", "draw"))).mean()

