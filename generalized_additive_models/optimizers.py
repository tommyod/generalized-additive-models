#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 06:22:30 2023

@author: tommy
"""
import numpy as np
import scipy as sp
from generalized_additive_models.utils import log
from sklearn.linear_model import Ridge
from sklearn.utils import Bunch

MACHINE_EPSILON = np.finfo(float).eps
EPSILON = np.sqrt(MACHINE_EPSILON)


class Optimizer:
    def _validate_params(self):
        pass

    def _validate_outputs(self):
        statistics_keys = ("", "", "", "")
        assert all((key in self._statistics.keys()) for key in statistics_keys)


class NaiveOptimizer(Optimizer):
    step_size = 0.99

    def __init__(self, *, X, D, y, link, distribution, max_iter, tol, beta=None):
        self.X = X
        self.D = D
        self.y = y
        self.link = link
        self.distribution = distribution
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta
        self.statistics_ = Bunch()

        log.info(f"Initialized {type(self).__name__}")

    def _validate_params(self):
        low, high = self.link.domain
        if np.any(self.y < high):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but largest y was: {self.y.max()}")

        if np.any(self.y > high):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but largest y was: {self.y.max()}")

    def initial_estimate(self):
        # Map the observations to the linear scale
        y_to_map = self.y.copy()

        # Numerical problems can occur if for instance we use the logit link
        # and y is 0 or 1, then we will map to infinity
        low, high = self.link.domain
        threshold = EPSILON**0.1
        y_to_map = np.maximum(np.minimum(y_to_map, high - threshold), low + threshold)
        assert np.all(y_to_map >= low + threshold)
        assert np.all(y_to_map <= high - threshold)

        mu_initial = self.link.link(y_to_map)
        assert np.all(np.isfinite(mu_initial))

        # Solve X @ beta = mu using Ridge regression
        ridge = Ridge(alpha=1e3, fit_intercept=False)
        ridge.fit(self.X, mu_initial)
        beta_initial = ridge.coef_
        # beta_initial, *_ = sp.linalg.lstsq(self.X, mu_initial)
        log.info(f"Initial beta estimate with norm: {sp.linalg.norm(beta_initial)}")
        return beta_initial

    def solve(self):
        num_observations, num_beta = self.X.shape

        beta = self.initial_estimate() if self.beta is None else self.beta

        # Step 1: Compute initial values
        eta = self.X @ beta
        mu = self.link.inverse_link(eta)
        log.info(f"Initial mu estimate in range: [{mu.min()}, {mu.max()}]")

        # List of betas to watch as optimization progresses
        betas = [beta]
        deviances = [self.distribution.deviance(y=self.y, mu=mu).mean()]

        alpha = 1  # Fisher weights, see page 250 in Wood, 2nd ed
        step_size = self.step_size

        # Set non-identifiable coefficients to zero
        # Compute Q R = A P
        # https://en.wikipedia.org/wiki/QR_decomposition#Column_pivoting
        Q, R, P = sp.linalg.qr(np.vstack((self.X, self.D)), mode="economic", pivoting=True)
        zero_coefs = (np.abs(np.diag(R)) < EPSILON)[P]
        X = self.X[:, ~zero_coefs]
        XT = X.T
        D = self.D[:, ~zero_coefs]
        DT = D.T
        betas[0] = betas[0][~zero_coefs]

        for iteration in range(1, self.max_iter + 1):
            log.info(f"Iteration {iteration}")

            # Step 1: Compute pseudodata z and iterative weights w
            z = self.link.derivative(mu) * (self.y - mu) / alpha + eta
            w = alpha / (self.link.derivative(mu) ** 2 * self.distribution.V(mu))

            # Step 3: Find beta to solve the weighted least squares objective
            # Solve f(z) = |z - X beta|^2_W + |D beta|^2

            # lhs = (self.X.T @ np.diag(w) @ self.X + self.D.T @ self.D)
            lhs = XT @ (w.reshape(-1, 1) * X) + DT @ D
            rhs = XT @ (w * z)
            beta_suggestion, *_ = sp.linalg.lstsq(lhs, rhs)

            # Increment the beta
            beta = step_size * beta_suggestion + betas[-1] * (1 - step_size)
            log.info(f"Beta estimate with norm: {sp.linalg.norm(beta)}")

            eta = X @ beta
            mu = self.link.inverse_link(eta)
            betas.append(beta)
            deviances.append(self.distribution.deviance(y=self.y, mu=mu).mean())

            if self._should_stop(betas):
                break

            if iteration > 1:
                step_size = step_size * 0.99
                log.info(f"Decreased step size to: {step_size:.6f}")
        else:
            log.warning(f"Solver did not converge within {iteration} iterations.")

        for i in range(len(betas)):
            beta_updated = np.zeros(num_beta, dtype=float)
            beta_updated[~zero_coefs] = beta
            betas[i] = beta_updated

        # Increase conditioning number
        X_T_X = self.X.T @ self.X
        X_T_X.flat[:: X_T_X.shape[0] + 1] += EPSILON

        # Compute degrees of freedom
        F = sp.linalg.solve(X_T_X + self.D.T @ self.D, X_T_X, assume_a="pos")
        A = np.linalg.multi_dot((self.X, sp.linalg.inv(X_T_X + self.D.T @ self.D), self.X.T))
        self.statistics_.edof = np.diag(F)

        # Compute generalized cross validation score
        # Equation (6.18) on page 260
        gcv = sp.linalg.norm(self.X @ betas[-1] - self.y) ** 2 * len(self.y) / (len(self.y) - np.trace(A)) ** 2
        self.statistics_.generalized_cross_validation_score = gcv

        # Compute the posterior distribution
        # See section 6.10 on page 293
        covariance = np.zeros(shape=(num_beta, num_beta), dtype=float)
        # https://numpy.org/doc/stable/reference/generated/numpy.ix_.html
        covariance[np.ix_(~zero_coefs, ~zero_coefs)] = sp.linalg.inv(lhs)
        self.statistics_.covariance = covariance

        return betas[-1]

    def _should_stop(self, betas):
        if len(betas) < 2:
            return False

        # Stopping criteria from
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet

        diffs = betas[-1] - betas[-2]
        assert np.all(np.isfinite(diffs))
        max_coord_update = np.max(np.abs(diffs))
        max_coord = np.max(np.abs(betas[-1]))
        log.info(f"Stopping criteria evaluation: {max_coord_update:.6f} <= {self.tol} * {max_coord:.6f} ")
        return max_coord_update / max_coord < self.tol


if __name__ == "__main__":
    from generalized_additive_models.gam import GAM
    from generalized_additive_models.terms import Spline, Intercept, Linear, TermList

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import pandas as pd

    # Get data as a DataFrame and Series
    data = fetch_california_housing(as_frame=True)
    df, y = data.data, data.target
    assert isinstance(df, pd.DataFrame)
    assert isinstance(y, pd.Series)

    columns_to_use = np.array([0, 3, 5])

    # Fit a model using DataFrame
    terms = TermList(Spline(c) for c in df.columns[columns_to_use])
    gam1 = GAM(terms=terms, fit_intercept=True)
    gam1.fit(df, y)

    # Fit a model using numpy array
    terms = TermList(Spline(c) for c in columns_to_use)
    gam2 = GAM(terms=terms, fit_intercept=True)
    gam2.fit(df.values, y.values)

    assert np.allclose(gam1.predict(df), gam2.predict(df.values))

    assert False

    X = np.exp(0.5 * np.random.randn(1000, 1))
    X = np.sort(X, axis=0)

    # X = np.linspace(0, 1*np.pi, num=2**10).reshape(-1, 1)**2
    y = (np.log1p(X) * np.sin(np.abs(X) ** 1.2)).ravel() + 10 + np.random.randn(len(X.ravel())) / (1 + X.ravel() ** 1)

    import matplotlib.pyplot as plt
    from generalized_additive_models.gam import GAM
    from generalized_additive_models.terms import Spline

    plt.scatter(X.ravel(), y)

    gam = GAM(terms=Spline(0, num_splines=9, degree=3, edges=None), fit_intercept=True)
    gam.fit(X, y)

    plt.plot(X, gam.predict(X), color="k")

    # plt.plot(X, gam.terms.fit_transform(X))

    plt.show()

    print("mean of y", np.mean(y))

    for term in gam.terms:
        print(term, term.coef_)
        print(term.coef_indicies_)

        X_smooth = np.linspace(np.min(X), np.max(X)).reshape(-1, 1)

        X_tilde = np.zeros_like(gam.terms.transform(X_smooth))
        X_tilde[:, term.coef_indicies_] = gam.terms.transform(X_smooth)[:, term.coef_indicies_]
        std = np.sqrt(np.diag(X_tilde @ gam.statistics_.covariance @ X_tilde.T))

        X_tilde = gam.terms.transform(X_smooth)[:, term.coef_indicies_]
        covar = gam.statistics_.covariance[np.ix_(term.coef_indicies_, term.coef_indicies_)]
        std2 = np.sqrt(np.sum((X_tilde @ covar) * X_tilde, axis=1))

        assert np.allclose(std, std2)

        y = term.transform(X_smooth) @ term.coef_
        plt.plot(X_smooth, y)
        plt.fill_between(X_smooth.ravel(), y - std, y + std, alpha=0.5)
        plt.show()
