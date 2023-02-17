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
    """The most straightforward and simple way to fit a GAM,
    ignoring almost all concerns about speed and numercial stability."""

    step_size = 1.0

    def __init__(self, *, X, D, y, link, distribution, max_iter, tol):
        self.X = X
        self.D = D
        self.y = y
        self.link = link
        self.distribution = distribution
        self.max_iter = max_iter
        self.tol = tol
        self.statistics_ = Bunch()

        log.info(f"Initialized {type(self).__name__}")

    def _validate_params(self):
        low, high = self.link.domain
        if np.any(self.y < high):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but largest y was: {self.y.max()}")

        if np.any(self.y > high):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but largest y was: {self.y.max()}")

    def _initial_estimate(self):
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

        beta = self._initial_estimate()

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
        log.info(f"Number of non-identifiable coefficients set to zero: {zero_coefs.sum()}")
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
            assert np.all(w > 0)

            # Step 3: Find beta to solve the weighted least squares objective
            # Solve f(z) = |z - X beta|^2_W + |D beta|^2

            # lhs = (self.X.T @ np.diag(w) @ self.X + self.D.T @ self.D)
            lhs = XT @ (w.reshape(-1, 1) * X) + DT @ D
            rhs = XT @ (w * z)
            beta_trial, *_ = sp.linalg.lstsq(lhs, rhs)

            # Increment the beta
            def objective_function(beta):
                return np.sum((z - X @ beta) ** 2 * w) + np.sum((D @ beta) ** 2)

            beta = betas[-1]
            current_objective = objective_function(beta)
            log.info(f"Objective function value: {current_objective}")
            for step_halfing in range(10):
                step_size = (1 / 2) ** step_halfing

                suggested_beta = step_size * beta_trial + (1 - step_size) * beta
                suggested_objective = objective_function(suggested_beta)
                log.info(f"Step size {step_size} gives objective {suggested_objective}")

                if suggested_objective <= current_objective:
                    log.info(f"Used step size: {step_size}")
                    beta = suggested_beta
                    break
            else:
                log.info("Step halving failed...")
                break

            # beta = step_size * beta_suggestion + betas[-1] * (1 - step_size)
            log.info(f"Beta estimate with norm: {sp.linalg.norm(beta)}")

            eta = X @ beta
            mu = self.link.inverse_link(eta)
            betas.append(beta)
            deviances.append(self.distribution.deviance(y=self.y, mu=mu).mean())

            if self._should_stop(betas):
                break
        else:
            log.warning(f"Solver did not converge within {iteration} iterations.")

        # Add zero coefficients back
        for i in range(len(betas)):
            beta_updated = np.zeros(num_beta, dtype=float)
            beta_updated[~zero_coefs] = beta
            betas[i] = beta_updated

        beta = betas[-1]

        # Compute the hat matrix: H = X @ (X.T @ W @ X + D.T @ D)^-1 @ W @ X.T
        # Also called the projection matrix or influence matrix (page 251 in Wood, 2nd ed)
        to_invert = self.X.T @ (w.reshape(-1, 1) * self.X) + self.D.T @ self.D
        to_invert.flat[:: to_invert.shape[0] + 1] += EPSILON  # Add to diagonal
        inverted = sp.linalg.inv(to_invert)

        # Only need the diagonal of H, so use the fact that
        # np.diag(A @ B @ A.T) = ((A @ B) * A).sum(axis=1)
        # to compute H below:
        # H = ((w.reshape(-1, 1) * self.X) @ inverted @ self.X.T)
        H_diag = np.sum(((w.reshape(-1, 1) * self.X) @ inverted) * self.X, axis=1)
        edof_per_coef = H_diag
        edof = edof_per_coef.sum()

        # Compute phi by equation (6.2) (page 251 in Wood, 2nd ed)
        # In the Gaussian case, phi is the variance
        if self.distribution.scale is not None:
            phi = self.distribution.scale
        else:
            phi = np.sum((z - self.X @ beta) ** 2 * w) / (len(beta) - edof)

        # Compute the covariance matrix of the parameters V_\beta (page 293 in Wood, 2nd ed)
        covariance = inverted * phi
        assert covariance.shape == (len(beta), len(beta))
        self.statistics_.covariance = covariance

        self.statistics_.edof_per_coef = edof_per_coef
        self.statistics_.edof = edof

        # Compute generalized cross validation score
        # Equation (6.18) on page 260
        # TODO: This is only the Gaussian case, see page 262
        gcv = sp.linalg.norm(self.X @ beta - self.y) ** 2 * len(self.y) / (len(self.y) - edof) ** 2
        self.statistics_.generalized_cross_validation_score = gcv

        return beta

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
    from generalized_additive_models.distributions import Binomial

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import pandas as pd

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data.data
    target = data.target

    gam = GAM(Spline(None), link="logit", distribution=Binomial(trials=1), max_iter=15)

    gam.fit(df, target)
