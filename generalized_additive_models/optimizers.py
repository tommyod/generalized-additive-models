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

            assert len(betas) >= 2
            error = sp.linalg.norm(betas[-1] - betas[-2]) / num_beta
            log.info(f"Error criterion |beta_n - beta_{{n-1}}|: {error:.6f}")
            log.info(f"Deviance: {deviances[-1]:.6f}")
            if error < self.tol:
                return betas[-1]

            if iteration > 1:
                previous_error = sp.linalg.norm(betas[-2] - betas[-3]) / num_beta
                if error > previous_error:
                    step_size = step_size * 0.99
                    log.info(f"Decreased step size to: {step_size:.6f}")

        log.warning(f"Solver did not converge within {iteration} iterations.")

        return betas[-1]

    def _should_stop(self):
        pass


if __name__ == "__main__":
    from generalized_additive_models.gam import GAM
    from generalized_additive_models.terms import Spline, Intercept
    import matplotlib.pyplot as plt

    # Poisson problem
    np.random.seed(1)
    x = np.linspace(0, 6 * np.pi, num=10_000)
    y = np.random.poisson(lam=(1.1 + np.sin(x)) * 10)
    X = x.reshape(-1, 1)

    poisson_gam = GAM(
        Spline(0, num_splines=10, degree=3, penalty=1, extrapolation="periodic"),
        link="log",
        distribution="poisson",
        max_iter=250,
    )
    poisson_gam.fit(X, y)

    plt.scatter(x, y)

    X_smooth = np.linspace(np.min(X), np.max(X), num=2**8).reshape(-1, 1)
    plt.plot(X_smooth, poisson_gam.predict(X_smooth), color="k", lw=3)

    plt.show()
    print(poisson_gam.coef_)

    # Logistic regression problem
    from generalized_additive_models.distributions import Binomial
    from scipy.special import logit, expit

    np.random.seed(1)
    x = np.linspace(0, 2 * np.pi, num=100)
    p = expit(np.sin(x) * 3)
    y = np.random.binomial(5, p, size=None)
    X = x.reshape(-1, 1)

    binomial_gam = GAM(
        Spline(0, num_splines=10, degree=3, penalty=1, extrapolation="periodic"),
        link="logit",
        distribution=Binomial(levels=5),
        max_iter=250,
        tol=1e-6,
    )
    binomial_gam.fit(X, y)

    plt.scatter(x, y + np.random.randn(len(y)) / 20)

    X_smooth = np.linspace(np.min(X), np.max(X), num=2**8).reshape(-1, 1)
    plt.plot(X_smooth, binomial_gam.predict(X_smooth), color="k", lw=3)

    plt.show()
    print(binomial_gam.coef_)
