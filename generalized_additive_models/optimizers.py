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

MACHINE_EPSILON = np.finfo(float).eps
EPSILON = np.sqrt(MACHINE_EPSILON)


class NaiveOptimizer:
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
        y_to_map = np.maximum(np.minimum(y_to_map, high - EPSILON), low + EPSILON)
        assert np.all(y_to_map >= low + EPSILON)
        assert np.all(y_to_map <= high - EPSILON)

        mu_initial = self.link.link(y_to_map)
        assert np.all(np.isfinite(mu_initial))

        # Solve X @ beta = mu using Ridge regression
        ridge = Ridge(alpha=1e3, fit_intercept=False)
        ridge.fit(self.X, mu_initial)
        beta_initial = ridge.coef_
        # beta_initial, *_ = sp.linalg.lstsq(self.X, mu_initial)
        log.info(f"Initial beta estimate: {beta_initial}")
        return beta_initial

    def solve(self):
        beta = self.initial_estimate() if self.beta is None else self.beta
        betas = [beta]

        # Step 1: Compute initial values
        eta = self.X @ beta
        mu = self.link.inverse_link(eta)
        log.info(f"Initial mu estimate: {mu}")

        # List of betas to watch as optimization progresses
        betas = [beta]

        alpha = 1  # Fisher weights, see page 250 in Wood, 2nd ed

        for iteration in range(1, self.max_iter + 1):
            log.info(f"Iteration {iteration}")

            # Step 1: Compute pseudodata z and iterative weights w
            z = self.link.gradient(mu) * (self.y - mu) / alpha + eta
            w = alpha / (self.link.gradient(mu) ** 2 * self.distribution.V(mu))

            # Step 3: Find beta to solve the weighted least squares objective
            # Solve |z - X beta|^2_W + |D beta|

            # lhs = (self.X.T @ np.diag(w) @ self.X + self.D.T @ self.D)
            lhs = self.X.T @ (w.reshape(-1, 1) * self.X) + self.D.T @ self.D
            rhs = self.X.T @ (w * z)
            beta_suggestion, *_ = sp.linalg.lstsq(lhs, rhs)

            # Increment the beta
            beta = self.step_size * beta_suggestion + betas[-1] * (1 - self.step_size)

            eta = self.X @ beta
            mu = self.link.inverse_link(eta)

            betas.append(beta)

            assert len(betas) >= 2
            error = sp.linalg.norm(betas[-1] - betas[-2])
            if error < self.tol:
                return betas[-1]

        return betas[-1]

    def _should_stop(self):
        pass
