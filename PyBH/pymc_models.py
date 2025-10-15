from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from arviz import r2_score
from causalpy.utils import round_num


class PyMCModel(pm.Model):
    """A wrapper class for PyMC models. This provides a scikit-learn like interface with
    methods like `fit`, `predict`, and `score`. It also provides other methods which are
    useful for causal inference.

    Example
    -------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> import pymc as pm
    >>> from causalpy.pymc_models import PyMCModel
    >>> class MyToyModel(PyMCModel):
    ...     def build_model(self, X, y, coords):
    ...         with self:
    ...             self.add_coords(coords)
    ...             X_ = pm.Data(name="X", value=X)
    ...             y_ = pm.Data(name="y", value=y)
    ...             beta = pm.Normal(
    ...                 "beta", mu=0, sigma=1, shape=(y.shape[1], X.shape[1])
    ...             )
    ...             sigma = pm.HalfNormal("sigma", sigma=1, shape=y.shape[1])
    ...             mu = pm.Deterministic(
    ...                 "mu", pm.math.dot(X_, beta.T), dims=["obs_ind", "treated_units"]
    ...             )
    ...             pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_)
    >>> rng = np.random.default_rng(seed=42)
    >>> X = xr.DataArray(
    ...     rng.normal(loc=0, scale=1, size=(20, 2)),
    ...     dims=["obs_ind", "coeffs"],
    ...     coords={"obs_ind": np.arange(20), "coeffs": ["coeff_0", "coeff_1"]},
    ... )
    >>> y = xr.DataArray(
    ...     rng.normal(loc=0, scale=1, size=(20, 1)),
    ...     dims=["obs_ind", "treated_units"],
    ...     coords={"obs_ind": np.arange(20), "treated_units": ["unit_0"]},
    ... )
    >>> model = MyToyModel(
    ...     sample_kwargs={
    ...         "chains": 2,
    ...         "draws": 2000,
    ...         "progressbar": False,
    ...         "random_seed": 42,
    ...     }
    ... )
    >>> model.fit(
    ...     X,
    ...     y,
    ...     coords={
    ...         "coeffs": ["coeff_0", "coeff_1"],
    ...         "obs_ind": np.arange(20),
    ...         "treated_units": ["unit_0"],
    ...     },
    ... )
    Inference data...
    >>> model.score(X, y)  # doctest: +ELLIPSIS
    unit_0_r2        ...
    unit_0_r2_std    ...
    dtype: float64
    >>> X_new = rng.normal(loc=0, scale=1, size=(20, 2))
    >>> model.predict(X_new)
    Inference data...
    """

    default_priors = {}

    def priors_from_data(self, X, y) -> Dict[str, Any]:
        """
        Generate priors dynamically based on the input data.

        This method allows models to set sensible priors that adapt to the scale
        and characteristics of the actual data being analyzed. It's called during
        the `fit()` method before model building, allowing data-driven prior
        specification that can improve model performance and convergence.

        The priors returned by this method are merged with any user-specified
        priors (passed via the `priors` parameter in `__init__`), with
        user-specified priors taking precedence in case of conflicts.

        Parameters
        ----------
        X : xarray.DataArray
            Input features/covariates with dimensions ["obs_ind", "coeffs"].
            Used to understand the scale and structure of predictors.
        y : xarray.DataArray
            Target variable with dimensions ["obs_ind", "treated_units"].
            Used to understand the scale and structure of the outcome.

        Returns
        -------
        Dict[str, Prior]
            Dictionary mapping parameter names to Prior objects. The keys should
            match parameter names used in the model's `build_model()` method.

        Notes
        -----
        The base implementation returns an empty dictionary, meaning no
        data-driven priors are set by default. Subclasses should override
        this method to implement data-adaptive prior specification.

        **Priority Order for Priors:**
        1. User-specified priors (passed to `__init__`)
        2. Data-driven priors (from this method)
        3. Default priors (from `default_priors` property)

        Examples
        --------
        A typical implementation might scale priors based on data variance:

        >>> def priors_from_data(self, X, y):
        ...     y_std = float(y.std())
        ...     return {
        ...         "sigma": Prior("HalfNormal", sigma=y_std, dims="treated_units"),
        ...         "beta": Prior(
        ...             "Normal",
        ...             mu=0,
        ...             sigma=2 * y_std,
        ...             dims=["treated_units", "coeffs"],
        ...         ),
        ...     }

        Or set shape parameters based on data dimensions:

        >>> def priors_from_data(self, X, y):
        ...     n_predictors = X.shape[1]
        ...     return {
        ...         "beta": Prior(
        ...             "Dirichlet",
        ...             a=np.ones(n_predictors),
        ...             dims=["treated_units", "coeffs"],
        ...         )
        ...     }

        See Also
        --------
        WeightedSumFitter.priors_from_data : Example implementation that sets
            Dirichlet prior shape based on number of control units.
        """
        return {}

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

        # Use integer indices for obs_ind to avoid datetime compatibility issues
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
        self.priors = {**self.priors_from_data(X, y), **self.priors}

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

    def calculate_impact(
        self, y_true: xr.DataArray, y_pred: az.InferenceData
    ) -> xr.DataArray:
        impact = y_true - y_pred["posterior_predictive"]["y_hat"]
        return impact.transpose(..., "obs_ind")

    def calculate_cumulative_impact(self, impact):
        return impact.cumsum(dim="obs_ind")

    def print_coefficients(self, labels, round_to=None) -> None:
        def print_row(
            max_label_length: int, name: str, coeff_samples: xr.DataArray, round_to: int
        ) -> None:
            """Print one row of the coefficient table"""
            formatted_name = f"  {name: <{max_label_length}}"
            formatted_val = f"{round_num(coeff_samples.mean().data, round_to)}, 94% HDI [{round_num(coeff_samples.quantile(0.03).data, round_to)}, {round_num(coeff_samples.quantile(1 - 0.03).data, round_to)}]"  # noqa: E501
            print(f"  {formatted_name}  {formatted_val}")

        def print_coefficients_for_unit(
            unit_coeffs: xr.DataArray,
            unit_sigma: xr.DataArray,
            labels: list,
            round_to: int,
        ) -> None:
            """Print coefficients for a single unit"""
            # Determine the width of the longest label
            max_label_length = max(len(name) for name in labels + ["y_hat_sigma"])

            for name in labels:
                coeff_samples = unit_coeffs.sel(coeffs=name)
                print_row(max_label_length, name, coeff_samples, round_to)

            # Add coefficient for measurement std
            print_row(max_label_length, "y_hat_sigma", unit_sigma, round_to)

        print("Model coefficients:")
        coeffs = az.extract(self.idata.posterior, var_names="beta")

        # Check if sigma or y_hat_sigma variable exists
        sigma_var_name = None
        if "sigma" in self.idata.posterior:
            sigma_var_name = "sigma"
        elif "y_hat_sigma" in self.idata.posterior:
            sigma_var_name = "y_hat_sigma"
        else:
            raise ValueError(
                "Neither 'sigma' nor 'y_hat_sigma' found in posterior"
            )  # pragma: no cover

        treated_units = coeffs.coords["treated_units"].values
        for unit in treated_units:
            if len(treated_units) > 1:
                print(f"\nTreated unit: {unit}")

            unit_coeffs = coeffs.sel(treated_units=unit)
            unit_sigma = az.extract(self.idata.posterior, var_names=sigma_var_name).sel(
                treated_units=unit
            )
            print_coefficients_for_unit(unit_coeffs, unit_sigma, labels, round_to or 2)
