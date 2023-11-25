#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:58:10 2023

@author: tommy


https://arxiv.org/pdf/1809.10632.pdf
"""

import copy
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.utils import Bunch, check_scalar
from sklearn.utils.validation import check_is_fitted

from generalized_additive_models.gam import GAM
from generalized_additive_models.terms import Categorical, Intercept, Linear, Spline, Tensor, Term
from generalized_additive_models.utils import cartesian


def model_checking(gam):
    # Common computations
    X, y, sample_weight = gam.X_, gam.y_, gam.sample_weight_
    predictions = gam.predict(X)
    residuals = y - predictions

    # Page 331 in Wood, 2nd ed

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

    # =========================================================================
    deviance = gam._distribution.deviance(y=y, mu=predictions, sample_weight=sample_weight, scaled=True)
    deviance_residuals = np.sign(residuals) * np.sqrt(deviance)

    sorted_deviance_residuals = np.sort(deviance_residuals)

    i = (np.arange(len(residuals)) + 0.5) / len(residuals)
    q_i = sp.stats.norm(loc=0, scale=1).ppf(i)

    ax1.scatter(q_i, sorted_deviance_residuals, s=2)

    min_value = min(np.min(q_i), sorted_deviance_residuals[0])
    max_value = max(np.max(q_i), sorted_deviance_residuals[-1])
    ax1.plot([min_value, max_value], [min_value, max_value], color="k")
    ax1.set_xlabel("Theoretical N(0, 1) quantiles")
    ax1.set_ylabel("Observed deviance residuals")

    # =========================================================================
    # ax2.set_title("Histogram of deviance residuals")
    # ax2.hist(deviance_residuals, bins="auto", density=True)
    # ax2.axvline(x=np.mean(deviance_residuals), color="k")
    # ax2.set_yticklabels([])

    ax2.set_title("Response vs. fitted values")
    deviance = gam._distribution.deviance(y=y, mu=predictions, scaled=False)
    deviance_residuals = np.sign(residuals) * np.sqrt(deviance)
    ax2.scatter(predictions, deviance_residuals, s=2)
    ax2.set_xlabel("Predicted values")
    ax2.set_ylabel("Deviance residuals")

    # =========================================================================
    ax3.set_title("Histogram of residuals")
    ax3.hist(residuals, bins="auto", density=True)
    ax3.axvline(x=np.mean(residuals), color="k")
    ax3.set_yticklabels([])

    # =========================================================================
    ax4.set_title("Response vs. fitted values")
    ax4.scatter(y, predictions, s=2)
    ax4.set_xlabel("Target values")
    ax4.set_ylabel("Predicted values")
    min_value = min(np.min(y), np.min(predictions))
    max_value = max(np.max(y), np.max(predictions))
    ax4.plot([min_value, max_value], [min_value, max_value], color="k")

    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, ls="--", zorder=0, alpha=0.33)

    fig.tight_layout()

    plt.show()

    # TODO:


def generate_X_grid(gam, term, X, *, extrapolation=0.01, num=100, meshgrid=True):
    if not isinstance(gam, GAM):
        raise TypeError(f"Parameter `gam` must be instance of GAM, found: {gam}")

    if not isinstance(term, Term):
        raise TypeError(f"Parameter `term` must be instance of Term, found: {term}")

    check_is_fitted(gam, attributes=["coef_"])
    check_is_fitted(term)

    if isinstance(term, Intercept):
        return np.array([[1.0]])

    if isinstance(term, (Linear, Spline)):
        X_data = term._get_column(X, selector="feature")

        min_, max_ = np.min(X_data), np.max(X_data)
        offset = (max_ - min_) * extrapolation
        return np.linspace(min_ - offset, max_ + offset, num=num).reshape(-1, 1)

    if isinstance(term, Categorical):
        return np.array(term.categories_).reshape(-1, 1)

    if isinstance(term, Tensor):
        linspaces = [generate_X_grid(gam, spline, X, extrapolation=extrapolation, num=num).ravel() for spline in term]

        # Return a cartesian grid (for predicting) and a meshgrid (for plotting)
        return (cartesian(linspaces), np.meshgrid(*linspaces, indexing="ij"))


def partial_effect(gam, term, standard_deviations=1.0, edges=None, linear_scale=True):
    """

    1 standard deviation  => 0.6827 coverage
    2 standard deviations => 0.9545 coverage
    3 standard deviations => 0.9973 coverage
    4 standard deviations => 0.9999 coverage

    Coverage of 1 standard deviation is given by:

    >>> from scipy.stats import norm
    >>> norm().cdf(1) - norm().cdf(-1)
    0.6826894921370...

    Parameters
    ----------
    term : Linear, Spline, str, int
        A term from the fitted gam (access via gam.terms), or a feature name
        corresponding to a term from the fitted gam.
    standard_deviations : float, optional
        The number of standard deviations to cover in the credible interval.
        The default is 1.0.

    Returns
    -------
    Bunch
        A dict-like object with results.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(100, 1))
    >>> y = np.sin(X).ravel() + rng.normal(scale=0.1, size=100)
    >>> spline = Spline(0)
    >>> gam = GAM(spline).fit(X, y)
    >>> results = partial_effect(gam, spline)

    Or loop through the model terms like so:

    >>> for term in gam.terms:
    ...     if isinstance(term, Intercept):
    ...         continue
    ...     results = partial_effect(gam, spline)
    ...     # Plotting code goes here

    """

    # ======================= PARAMETER VALIDATION ============================
    if not isinstance(gam, GAM):
        raise TypeError(f"Parameter `gam` must be instance of GAM, found: {gam}")

    if not isinstance(term, Term):
        raise TypeError(f"Parameter `term` must be instance of Term, found: {term}")

    check_is_fitted(gam, attributes=["coef_"])
    check_is_fitted(term)

    if term not in gam.terms:
        raise ValueError(f"Term not found in model: {term}")

    standard_deviations = check_scalar(
        standard_deviations,
        "standard_deviations",
        target_type=Real,
        min_val=0,
        include_boundaries="neither",
    )

    # If the term is a number or string, try to fetch it from the terms
    if isinstance(term, (str, Integral)):
        try:
            term = next(t for t in gam.terms if t.feature == term)
        except StopIteration:
            return ValueError(f"Could not find term with feature: {term}")

    # ================================ LOGIC  =================================

    # Get data related to term and create a smooth grid
    term = copy.deepcopy(term)  # Copy so feature_ is not changed by term.transform() below
    data = term._get_column(gam.X_, selector="feature")

    X_original_transformed = term.transform(gam.X_)

    meshgrid = None
    if isinstance(term, Tensor):
        X_smooth, meshgrid = generate_X_grid(gam, term, gam.X_, extrapolation=0.01, num=100)
        for i, spline in enumerate(term):
            spline.set_params(feature=i)
    else:
        X_smooth = generate_X_grid(gam, term, gam.X_, extrapolation=0.01, num=100)
        term.set_params(feature=0)

    # Predict on smooth grid
    X = term.transform(X_smooth)
    linear_predictions = X @ term.coef_
    predictions = linear_predictions  # if linear_scale else gam._link.inverse_link(linear_predictions)

    # Get the covariance matrix associated with the coefficients of this term
    V = gam.results_.covariance[np.ix_(term.coef_idx_, term.coef_idx_)]

    # The variance of y = X @ \beta is given by diag(X @ V @ X.T)
    # Page 293 in Wood, or see: https://math.stackexchange.com/a/2365257
    # Since we only need the diagonal, we don't form the full matrix product
    # Also, see equation (375) in The Matrix Cookbook
    # https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf

    stdev_array = np.sqrt(np.sum((X @ V) * X, axis=1))
    assert np.all(stdev_array > 0)
    assert np.allclose(stdev_array**2, np.diag(X @ V @ X.T))

    # For partial residual plots
    # https://en.wikipedia.org/wiki/Partial_residual_plot#Definition
    residuals = gam.y_ - gam._link.inverse_link(gam.model_matrix_ @ gam.coef_)

    # Prepare the linear results
    result = Bunch(
        x=X_smooth,
        y=predictions,
        y_low=predictions - standard_deviations * stdev_array,
        y_high=predictions + standard_deviations * stdev_array,
        x_obs=data,
        y_partial_residuals=(X_original_transformed @ term.coef_) + residuals,
        meshgrid=meshgrid,
    )

    # Map the variables
    if not linear_scale:
        result.y = gam._link.inverse_link(result.y)
        result.y_low = gam._link.inverse_link(result.y_low)
        result.y_high = gam._link.inverse_link(result.y_high)
        model_predictions = gam._link.inverse_link(X_original_transformed @ term.coef_)
        result.y_partial_residuals = (model_predictions + residuals,)

    return result


def plot_qq(gam, return_data=False):
    # From paper: "On quantile quantile plots for generalized linear models"
    # By Nicole H. Augustin, Erik-André Sauleau, Simon N. Wood
    # https://www.sciencedirect.com/science/article/pii/S0167947312000692

    # Paper
    X, y, sample_weight = gam.X_, gam.y_, gam.sample_weight_

    # Compute deviance residuals, following the notation in Section 2.1
    mu = gam.predict(X)
    deviance = gam._distribution.deviance(y=y, mu=mu, sample_weight=sample_weight, scaled=True)
    d_i = np.sort(np.sign(y - mu) * np.sqrt(deviance))

    # Number of simulations
    simulations = int(max(10**5 / len(y), 25))

    # Create arrays of size (simulations, num_samples)
    simulated_y = gam.sample(y, size=(simulations, len(y)))
    predictions = np.outer(np.ones(simulations), gam.predict(X))
    sample_weight = np.outer(np.ones(simulations), sample_weight)

    # Compute deviance residuals for all simulations
    # This gives a probability distribution for the deviance residuals
    deviance = gam._distribution.deviance(y=simulated_y, mu=predictions, sample_weight=sample_weight, scaled=True)
    deviance_residuals = np.sign(y - predictions) * np.sqrt(deviance)

    # Compute percentiles => approx the inverse CDF of the residual distribution
    i = (np.arange(len(y)) + 0.5) / len(y)
    d_star_i = np.percentile(deviance_residuals, q=i * 100)

    if return_data:
        return Bunch(
            obs_deviance_residuals=d_i,
            theoretical_deviance_residuals=d_star_i,
            simulated_deviance_residuals=deviance_residuals,
        )

    # Create the figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(d_star_i, d_i, s=5, zorder=15)

    min_value = min(np.min(d_i), np.min(d_star_i))
    max_value = max(np.max(d_i), np.max(d_star_i))
    ax.plot([min_value, max_value], [min_value, max_value], color="black", zorder=10)

    deviance_residuals = np.sort(deviance_residuals, axis=1)
    low, high = np.percentile(deviance_residuals, q=[1, 99], axis=0)

    ax.fill_between(d_star_i, low, high, alpha=0.33, color="black", zorder=5)

    ax.set_ylabel("Observed deviance residuals")
    ax.set_xlabel("Theoretical quantiles")
    ax.grid(True, ls="--", zorder=0, alpha=0.33)
    fig.tight_layout()


if __name__ == "__main__":
    import random

    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df, y = data.data, data.target * 100

    df = df.assign(cat=[random.choice(list("abcdef")) for _ in range(len(df))])

    # Fit a model using column names
    gam = GAM(
        terms=Spline("AveRooms")
        + Linear("Population")
        + Tensor([Spline("Latitude"), Spline("Longitude")])
        + Categorical("cat"),
        fit_intercept=True,
        # link="log"
    )
    gam.fit(df, y)

    plot_qq(gam)
    1 / 0

    a, b = generate_X_grid(gam, gam.terms[2], df, num=10)

    model_checking(gam)

    # Paper
    X, y, sample_weight = gam.X_, gam.y_, gam.sample_weight_

    # Compute deviance residuals
    mu = gam.predict(X)
    deviance = gam._distribution.deviance(y=y, mu=mu, sample_weight=sample_weight, scaled=True)
    d_i = np.sign(y - mu) * np.sqrt(deviance)
    d_i = np.sort(d_i)

    simulations = 1000
    simulated_y = gam.sample(y, size=(simulations, len(y)))
    predictions = np.outer(np.ones(simulations), gam.predict(X))
    sample_weight = np.outer(np.ones(simulations), sample_weight)
    assert simulated_y.shape == predictions.shape

    # residuals = simulated_y - gam.predict(X)

    deviance = gam._distribution.deviance(y=simulated_y, mu=predictions, sample_weight=sample_weight, scaled=True)
    deviance_residuals = np.sign(y - predictions) * np.sqrt(deviance)

    i = (np.arange(len(y)) + 0.5) / len(y)
    d_star_i = np.percentile(deviance_residuals, q=i * 100)

    plt.figure()
    plt.scatter(d_i, d_star_i, s=5)

    min_value = min(np.min(d_i), np.min(d_star_i))
    max_value = max(np.max(d_i), np.max(d_star_i))
    plt.plot([min_value, max_value], [min_value, max_value], color="k")

    deviance_residuals = np.sort(deviance_residuals, axis=1)
    low, high = np.percentile(deviance_residuals, q=[1, 99], axis=0)

    plt.fill_between(d_star_i, low, high, alpha=0.33, color="black")
