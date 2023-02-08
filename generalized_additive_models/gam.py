#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:05:41 2023

@author: tommy
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.utils._param_validation import Hidden, Interval, StrOptions
from numbers import Real, Integral
from generalized_additive_models.terms import Term, Spline, Linear, TermList, Intercept, Tensor
from generalized_additive_models.links import LINKS
from generalized_additive_models.distributions import DISTRIBUTIONS
from generalized_additive_models.optimizers import NaiveOptimizer

# from generalized_additive_models.distributions import DISTRIBUTIONS


class GAM(BaseEstimator):
    _parameter_constraints: dict = {
        "terms": [Term, TermList],
        "distribution": [StrOptions({"normal", "poisson", "gamma"})],
        "link": [StrOptions({"identity", "log"})],
        "fit_intercept": ["boolean"],
        "solver": [
            StrOptions({"lbfgs", "newton-cholesky"}),
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
        solver="lbfgs",
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

    def _validate_params(self):
        super()._validate_params()
        self._link = LINKS[self.link]()
        self._distribution = DISTRIBUTIONS[self.distribution]()
        self.terms = TermList(self.terms)

        if self.fit_intercept and Intercept() not in self.terms:
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
        self._validate_params()
        X, y = self._validate_data(
            X,
            y,
            dtype=[np.float64, np.float32],
            y_numeric=True,
            multi_output=False,
        )

        self.model_matrix_ = self.terms.fit_transform(X)
        penalty_matrix = self.terms.penalty_matrix()

        optimizer = NaiveOptimizer(
            X=self.model_matrix_,
            D=self.terms.penalty_matrix(),
            y=y,
            link=self._link,
            distribution=self._distribution,
            max_iter=self.max_iter,
            tol=self.tol,
            beta=None,
        )

        self.coef_ = optimizer.solve()

        if False:
            lhs = self.model_matrix_.T @ self.model_matrix_ + penalty_matrix.T @ penalty_matrix
            rhs = self.model_matrix_.T @ y

            self.coef_, *_ = sp.linalg.lstsq(
                lhs,
                rhs,
            )

        # self.coef_ = np.zeros(sum(term.num_coefficients for term in self.terms))
        # self.coef_ = np.linalg.lstsq(lhs, rhs)

        return self

    def predict(self, X):
        model_matrix = self.terms.transform(X)
        return self._link.inverse_link(model_matrix @ self.coef_)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


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
x = np.linspace(0, 2 * np.pi, num=100)
y = np.random.poisson(lam=1.1 + np.sin(x))
X = x.reshape(-1, 1)

poisson_gam = GAM(
    Spline(0, num_splines=5, degree=3, penalty=0.01, extrapolation="periodic"),
    link="log",
    distribution="poisson",
    max_iter=25,
)
poisson_gam.fit(X, y)

plt.scatter(x, y)

X_smooth = np.linspace(np.min(X), np.max(X), num=2**8).reshape(-1, 1)
plt.plot(X_smooth, poisson_gam.predict(X_smooth), color="k")


if __name__ == "__main__":
    import pytest

    # pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])
