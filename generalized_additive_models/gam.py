#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:05:41 2023

@author: tommy
"""

import copy
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch, check_scalar
from sklearn.utils._param_validation import Hidden, Interval, StrOptions

# https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/utils/validation.py#L1870
from sklearn.utils.validation import check_is_fitted

from generalized_additive_models.distributions import DISTRIBUTIONS, Distribution
from generalized_additive_models.links import LINKS, Link
from generalized_additive_models.optimizers import PIRLS
from generalized_additive_models.terms import Intercept, Linear, Spline, Term, TermList

# from generalized_additive_models.distributions import DISTRIBUTIONS


class GAM(BaseEstimator):
    """Generalized Additive Model.

    Examples
    --------
    >>> from generalized_additive_models import GAM, Spline, Categorical
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes(as_frame=True)
    >>> df = data.data
    >>> y = data.target
    >>> gam = GAM(Spline("age") + Spline("bmi") + Spline("bp") + Categorical("sex"))
    >>> gam = gam.fit(df, y)
    >>> predictions = gam.predict(df)
    >>> for term in gam.terms:
    ...     print(term, term.coef_) # doctest: +SKIP

    """

    _parameter_constraints: dict = {
        "terms": [Term, TermList],
        "distribution": [StrOptions(set(DISTRIBUTIONS.keys())), Distribution],
        "link": [StrOptions(set(LINKS.keys())), Link],
        "fit_intercept": ["boolean"],
        "solver": [
            StrOptions({"pirls"}),
            Hidden(type),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "warm_start": ["boolean"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        terms=None,
        *,
        distribution="normal",
        link="identity",
        fit_intercept=True,
        solver="pirls",
        max_iter=100,
        tol=0.0001,
        warm_start=False,
        verbose=0,
    ):
        self.terms = terms
        self.distribution = distribution
        self.link = link
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def _validate_params(self, X):
        super()._validate_params()
        self._link = LINKS[self.link]() if isinstance(self.link, str) else self.link
        self._distribution = (
            DISTRIBUTIONS[self.distribution]() if isinstance(self.distribution, str) else self.distribution
        )

        if self.solver == "pirls":
            self._solver = PIRLS
        else:
            raise ValueError("Unknown solver.")

        self.terms = TermList(self.terms)

        # Auto model
        # If only a single Term is passed, and that term has `feature=None`,
        # then expand and use one term per column with the other parameters
        num_samples, num_features = X.shape
        if len(self.terms) == 1:
            term = self.terms[0]
            if isinstance(term, Spline) and term.feature is None:
                term_params = term.get_params()
                term_params.pop("feature")
                self.terms = TermList([Spline(feature=i, **term_params) for i in range(num_features)])
            elif isinstance(term, Linear) and term.feature is None:
                term_params = term.get_params()
                term_params.pop("feature")
                self.terms = TermList([Linear(feature=i, **term_params) for i in range(num_features)])

        if self.fit_intercept and (Intercept() not in self.terms):
            self.terms.append(Intercept())

    def fit(self, X, y, sample_weight=None):
        """Fit model to data.


        Examples
        --------

        >>> rng = np.random.default_rng(32)
        >>> X = rng.normal(size=(100, 1))
        >>> y = np.sin(X).ravel()
        >>> gam = GAM(Spline(0))
        >>> gam.fit(X, y)
        GAM(terms=TermList(data=[Spline(feature=0), Intercept()]))

        """
        self._validate_params(X)

        self.model_matrix_ = self.terms.fit_transform(X)
        # =============================================================================
        #         X, y = self._validate_data(
        #             X,
        #             y,
        #             dtype=[np.float64, np.float32],
        #             y_numeric=True,
        #             multi_output=False,
        #         )
        # =============================================================================
        self.X_ = X.copy()  # Store a copy used for patial effects
        self.y_ = y.copy()

        optimizer = self._solver(
            X=self.model_matrix_,
            D=self.terms.penalty_matrix(),
            y=y,
            link=self._link,
            distribution=self._distribution,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )

        # Copy over solver information
        self.coef_ = optimizer.solve().copy()
        self.statistics_ = copy.deepcopy(optimizer.statistics_)

        # Assign coefficients to terms
        coef_idx = 0
        for term in self.terms:
            term.coef_ = self.coef_[coef_idx : coef_idx + term.num_coefficients]
            term.coef_indicies_ = np.arange(coef_idx, coef_idx + term.num_coefficients)
            coef_idx += term.num_coefficients
            assert len(term.coef_) == term.num_coefficients
        assert sum(len(term.coef_) for term in self.terms) == len(self.coef_)

        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["coef_"])

        model_matrix = self.terms.transform(X)
        return self._link.inverse_link(model_matrix @ self.coef_)

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self, attributes=["coef_"])

        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def partial_effect(self, term, standard_deviations=1.0, edges=None):
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
        >>> results = gam.partial_effect(spline)

        Alternatively, use the feature index or name:

        >>> results = gam.partial_effect(0)

        Or loop through the model terms like so:

        >>> for term in gam.terms:
        ...     if isinstance(term, Intercept):
        ...         continue
        ...     results = gam.partial_effect(term)
        ...     # Plotting code goes here

        """
        check_is_fitted(self, attributes=["coef_"])
        standard_deviations = check_scalar(
            standard_deviations, "standard_deviations", target_type=Real, min_val=0, include_boundaries="neither"
        )

        # If the term is a number or string, try to fetch it from the terms
        if isinstance(term, (str, Integral)):
            try:
                term = next(t for t in self.terms if t.feature == term)
            except StopIteration:
                return ValueError(f"Could not find term with feature: {term}")

        elif isinstance(term, (Linear, Spline)):
            if term not in self.terms:
                raise ValueError(f"Term not found in model: {term}")

        else:
            raise TypeError(f"`term` must be of type Linear or Spline, but found: {term}")

        # Get data related to term and create a smooth grid
        term = copy.deepcopy(term)  # Copy so feature_ is not changed by term.transform() below
        data = term._get_column(self.X_, selector="feature")
        if edges is None:
            min_val, max_val = np.min(data), np.max(data)
            range_ = max_val - min_val
            min_val, max_val = min_val - 0.01 * range_, max_val + 0.01 * range_
        else:
            min_val, max_val = edges

        X_smooth = np.linspace(min_val, max_val, num=2**10)

        # Predict on smooth grid
        X = term.transform(X_smooth.reshape(-1, 1))
        predictions = X @ term.coef_

        # Get the covariance matrix associated with the coefficients of this term
        V = self.statistics_.covariance[np.ix_(term.coef_indicies_, term.coef_indicies_)]

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
        residuals = self.y_ - (self.model_matrix_ @ self.coef_)

        # Prepare the results
        return Bunch(
            x=X_smooth,
            y=predictions,
            y_low=predictions - standard_deviations * stdev_array,
            y_high=predictions + standard_deviations * stdev_array,
            x_obs=data,
            y_partial_residuals=(term.transform(data.reshape(-1, 1)) @ term.coef_) + residuals,
        )


if __name__ == "__main__":
    if False:
        # Create a data set which is hard for an additive model to predict
        rng = np.random.default_rng(42)
        X = rng.triangular(0, mode=0.5, right=1, size=(1000, 1))
        X = np.sort(X, axis=0)

        y = X.ravel() ** 2 + rng.normal(scale=0.05, size=(1000))

        gam = GAM(terms=Spline(0, penalty=1, num_splines=3, degree=0) + Intercept(), fit_intercept=False)
        # gam = GAM(terms=Linear(0) + Intercept(), fit_intercept=False)

        gam.fit(X, y)

        print("mean data value", y.mean())
        print("intercept of model", gam.coef_[-1])

        x_smooth = np.linspace(0, 1, num=100).reshape(-1, 1)
        preds = gam.predict(np.linspace(0, 1, num=100).reshape(-1, 1))

        plt.figure()
        plt.scatter(X, y)
        plt.plot(np.linspace(0, 1, num=100).reshape(-1, 1), preds, color="k")
        plt.show()

        # Poisson problem
        np.random.seed(1)
        x = np.linspace(0, 2 * np.pi, num=10_000)
        lambda_ = 3 + np.sin(x) * 2
        y = np.random.poisson(lam=lambda_)
        X = x.reshape(-1, 1)

        poisson_gam = GAM(
            Spline(0, num_splines=5, degree=3, penalty=1, extrapolation="periodic"),
            link="log",
            distribution="poisson",
            max_iter=25,
        )
        poisson_gam.fit(X, y)

        plt.scatter(x, y)
        plt.plot(X.ravel(), lambda_, color="red")

        X_smooth = np.linspace(np.min(X), np.max(X), num=2**8).reshape(-1, 1)
        plt.plot(X_smooth, poisson_gam.predict(X_smooth), color="k")


if __name__ == "__main__":
    import pytest

    # pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])

    X = np.linspace(0.5, 2 * np.pi - 0.5, num=99).reshape(-1, 1)
    y = np.sin(X.ravel()) + np.random.randn(99) / 10

    # for num_splines in range(5, 300, 5):
    for penalty in np.logspace(-2, 4, num=100):
        gam = GAM(Spline(0, num_splines=20, penalty=penalty))
        gam.fit(X, y)

        plt.title(penalty)
        plt.scatter(X, y)

        plt.plot(X, gam.predict(X), color="black", lw=3)

        plt.show()
