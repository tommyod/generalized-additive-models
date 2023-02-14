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


class NaiveOptimizer(Optimizer):
    step_size = 0.9

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
        betas = [beta]

        # Step 1: Compute initial values
        eta = self.X @ beta
        mu = self.link.inverse_link(eta)
        log.info(f"Initial mu estimate in range: [{mu.min()}, {mu.max()}]")

        # List of betas to watch as optimization progresses
        betas = [beta]
        deviances = [self.distribution.deviance(y=self.y, mu=mu).mean()]

        alpha = 1  # Fisher weights, see page 250 in Wood, 2nd ed
        step_size = self.step_size

        for iteration in range(1, self.max_iter + 1):
            log.info(f"Iteration {iteration}")

            # Step 1: Compute pseudodata z and iterative weights w
            z = self.link.gradient(mu) * (self.y - mu) / alpha + eta
            w = alpha / (self.link.gradient(mu) ** 2 * self.distribution.V(mu))

            # Step 3: Find beta to solve the weighted least squares objective
            # Solve f(z) = |z - X beta|^2_W + |D beta|^2

            # lhs = (self.X.T @ np.diag(w) @ self.X + self.D.T @ self.D)
            lhs = self.X.T @ (w.reshape(-1, 1) * self.X) + self.D.T @ self.D
            rhs = self.X.T @ (w * z)
            beta_suggestion, *_ = sp.linalg.lstsq(lhs, rhs)

            # Increment the beta
            beta = step_size * beta_suggestion + betas[-1] * (1 - step_size)
            log.info(f"Beta estimate with norm: {sp.linalg.norm(beta)}")

            eta = self.X @ beta
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

        # Increase conditioning number
        X_T_X = self.X.T @ self.X
        X_T_X.flat[:: X_T_X.shape[0] + 1] += EPSILON

        F = sp.linalg.solve(X_T_X + self.D.T @ self.D, X_T_X, assume_a="pos")
        A = np.linalg.multi_dot((self.X, sp.linalg.inv(X_T_X + self.D.T @ self.D), self.X.T))
        self.statistics_.edof = np.diag(F)

        # Equation (6.18) on page 260
        gcv = sp.linalg.norm(self.X @ betas[-1] - self.y) ** 2 * len(self.y) / (len(self.y) - np.trace(A)) ** 2
        self.statistics_.generalized_cross_validation_score = gcv

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
    pass
