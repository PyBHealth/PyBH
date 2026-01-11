from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import r2_score


class PyMCModel(pm.Model):
    """A wrapper class for PyMC models. This provides a scikit-learn like interface with
    methods like `fit`, `predict`, and `score`. It also provides other methods which are
    useful for causal inference.
    """

    default_priors = {}

    def __init__(
        self,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        priors: dict[str, Any] | None = None,
    ):
        """
        :param sample_kwargs: A dictionary of kwargs that get unpacked and passed to the
            :func:`pymc.sample` function. Defaults to an empty dictionary.
        """
        super().__init__()
        self.idata = None
        self.sample_kwargs = sample_kwargs if sample_kwargs is not None else {}

        self.priors = {**self.default_priors, **(priors or {})}

    def build_model(self, X, y, coords) -> None:
        """Build the model, must be implemented by subclass."""
        raise NotImplementedError(
            "This method must be implemented by a subclass"
        )  # pragma: no cover

    def _data_setter(self, X: xr.DataArray) -> None:
        """
        Set data for the model.

        This method is used internally to register new data for the model for
        prediction.

        NOTE: We are actively changing the `X`. Often, this matrix will have a different
        number of rows than the original data. So to make the shapes work, we need to
        update all data nodes in the model to have the correct shape. The values are not
        used, so we set them to 0. In our case, we just have data nodes X and y, but if
        in the future we get more complex models with more data nodes, then we'll need
        to update all of them - ideally programmatically.
        """
        new_no_of_observations = X.shape[0]

        # Use integer indices for obs_ind to avoid
        # datetime compatibility issues with PyMC
        obs_coords = np.arange(new_no_of_observations)

        with self:
            # Get the number of treated units from the model coordinates
            treated_units_coord = getattr(self, "coords", {}).get(
                "treated_units", ["unit_0"]
            )
            n_treated_units = len(treated_units_coord)

            # Always use 2D format for consistency
            pm.set_data(
                {"X": X, "y": np.zeros((new_no_of_observations, n_treated_units))},
                coords={"obs_ind": obs_coords},
            )

    def fit(self, X, y, coords: Optional[Dict[str, Any]] = None) -> None:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions, placing them in the model's idata attribute.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)

        # Merge priors with precedence: user-specified > data-driven > defaults
        # Data-driven priors are computed first,
        # then user-specified priors override them
        self.priors = {**self.priors}

        self.build_model(X, y, coords)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata, progressbar=False, random_seed=random_seed
                )
            )
        return self.idata

    def predict(self, X: xr.DataArray):
        """
        Predict data given input data `X`

        .. caution::
            Results in KeyError if model hasn't been fit.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        random_seed = self.sample_kwargs.get("random_seed", None)
        self._data_setter(X)
        with self:
            pp = pm.sample_posterior_predictive(
                self.idata,
                var_names=["y_hat", "mu"],
                progressbar=False,
                random_seed=random_seed,
            )

        # Assign coordinates from input X to ensure xarray operations work correctly
        # This is necessary because PyMC uses integer indices internally, but we need
        # to preserve the original coordinates (e.g., datetime indices) for proper
        # alignment with other xarray operations like calculate_impact()
        if isinstance(X, xr.DataArray) and "obs_ind" in X.coords:
            pp["posterior_predictive"] = pp["posterior_predictive"].assign_coords(
                obs_ind=X.obs_ind
            )

        return pp

    def score(self, X: xr.DataArray, y: xr.DataArray) -> pd.Series:
        """Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.

        Note that the score is based on a comparison of the observed data ``y`` and the
        model's expected value of the data, `mu`.

        .. caution::

            The Bayesian :math:`R^2` is not the same as the traditional coefficient of
            determination, https://en.wikipedia.org/wiki/Coefficient_of_determination.

        """
        mu = self.predict(X)
        mu_data = az.extract(mu, group="posterior_predictive", var_names="mu")

        scores = {}

        # Always iterate over treated_units dimension - no branching needed!
        for i, unit in enumerate(mu_data.coords["treated_units"].values):
            unit_mu = mu_data.sel(treated_units=unit).T  # (sample, obs_ind)
            unit_y = y.sel(treated_units=unit).data
            unit_score = r2_score(unit_y, unit_mu.data)
            scores[f"unit_{i}_r2"] = unit_score["r2"]
            scores[f"unit_{i}_r2_std"] = unit_score["r2_std"]

        return pd.Series(scores)


class LinearRegression(PyMCModel):
    r"""
    Custom PyMC model for linear regression.

    Defines the PyMC model

    .. math::
        \beta &\sim \mathrm{Normal}(0, 50) \\
        \sigma &\sim \mathrm{HalfNormal}(1) \\
        \mu &= X \cdot \beta \\
        y &\sim \mathrm{Normal}(\mu, \sigma) \\
    """  # noqa: W605

    def build_model(self, X, y, coords):
        """
        Defines the PyMC model
        """
        with self:
            # Ensure treated_units coordinate exists for consistency
            if "treated_units" not in coords:
                coords = coords.copy()
                coords["treated_units"] = ["unit_0"]

            self.add_coords(coords)
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y, dims=["obs_ind", "treated_units"])
            beta = pm.Normal("beta", mu=0, sigma=50, dims=["treated_units", "coeffs"])
            mu = pm.Deterministic(
                "mu", pt.dot(X, beta.T), dims=["obs_ind", "treated_units"]
            )
            sigma = pm.HalfNormal("sigma", sigma=5, dims=["treated_units"])
            y_hat = pm.Normal(  # noqa: F841
                "y_hat",
                mu=mu,
                sigma=sigma,
                observed=y,
                dims=["obs_ind", "treated_units"],
            )
