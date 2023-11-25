#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


class QQDisplay:
    def __init__(self, *, residuals, quantiles, residuals_low=None, residuals_high=None):
        self.residuals = residuals
        self.quantiles = quantiles
        self.residuals_low = residuals_low
        self.residuals_high = residuals_high

    def plot(
        self,
        ax=None,
        *,
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        if scatter_kwargs is None:
            scatter_kwargs = {}

        if line_kwargs is None:
            line_kwargs = {}

        default_scatter_kwargs = {"alpha": 0.8, "s": 15}
        scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}

        default_line_kwargs = {"color": "black", "alpha": 0.7, "linestyle": "--"}
        line_kwargs = {**default_line_kwargs, **line_kwargs}

        if ax is None:
            _, ax = plt.subplots()

        self.scatter_ = ax.scatter(self.quantiles, self.residuals, **scatter_kwargs)

        # Compute line
        min_value = max(np.min(self.quantiles), np.min(self.residuals))
        max_value = min(np.max(self.quantiles), np.max(self.residuals))
        ax.line_ = ax.plot([min_value, max_value], [min_value, max_value], **line_kwargs)

        if self.residuals_low is not None and self.residuals_high is not None:
            self.fill_between_ = ax.fill_between(self.quantiles, self.residuals_low, self.residuals_high, alpha=0.5)

        ax.set_xlabel("Theoretical N(0, 1) quantiles")
        ax.set_ylabel("Observed residuals")

        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    def from_estimator(
        cls,
        gam,
        X,
        y,
        *,
        residuals="deviance",
        standardized=True,
        method="normal",
        ax=None,
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        if method not in ("normal", "simulate"):
            raise ValueError("Parameter `method` must be in 'normal' or 'simulate'")

        y = np.array(y, dtype=float)

        if method == "normal":
            residuals = np.sort(gam.residuals(X, y, residuals=residuals, standardized=standardized))

            # Loop up theoretical quantiles
            i = (np.arange(len(residuals)) + 0.5) / len(residuals)
            quantiles = sp.stats.norm(loc=0, scale=1).ppf(i)

            viz = cls(
                residuals=residuals,
                quantiles=quantiles,
            )

            return viz.plot(ax=ax, scatter_kwargs=scatter_kwargs, line_kwargs=line_kwargs)

        elif method == "simulate":
            # From paper: "On quantile quantile plots for generalized linear models"
            # By Nicole H. Augustin, Erik-André Sauleau, Simon N. Wood
            # https://www.sciencedirect.com/science/article/pii/S0167947312000692

            # Number of simulations
            simulations = int(max(10**5 / len(y), 25))

            # Create arrays of size (simulations, num_samples)
            y_pred = gam.predict(X)
            predictions = np.outer(np.ones(simulations), y_pred)
            simulated_y = gam.sample(y_pred, size=(simulations, len(y)))

            # Compute deviance residuals for all simulations
            # This gives a probability distribution for the deviance residuals
            deviance = gam._distribution.deviance(y=simulated_y, mu=predictions, sample_weight=None, scaled=True)
            deviance_residuals = np.sign(y - predictions) * np.sqrt(deviance)

            # Compute percentiles => approx the inverse CDF of the residual distribution
            i = (np.arange(len(y)) + 0.5) / len(y)
            d_star_i = np.percentile(deviance_residuals, q=i * 100)

            deviance_residuals = np.sort(deviance_residuals, axis=1)
            residuals = np.sort(gam.residuals(X, y, residuals=residuals, standardized=standardized))
            low, _, high = np.percentile(deviance_residuals, q=[1, 50, 99], axis=0)

            viz = cls(
                residuals=residuals,
                quantiles=d_star_i,  # x-axis
                residuals_low=low,
                residuals_high=high,
            )

            return viz.plot(ax=ax, scatter_kwargs=scatter_kwargs, line_kwargs=line_kwargs)
