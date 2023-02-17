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
        threshold = EPSILON**0.25
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

        def objective_function_value(X, D, beta, y):
            mu = self.link.inverse_link(X @ beta)
            deviance = self.distribution.deviance(y=y, mu=mu, scaled=True).sum()
            penalty = sp.linalg.norm(D @ beta) ** 2
            return deviance + penalty

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
            beta_previous = betas[-1]
            objective_previous = objective_function_value(X, D, beta_previous, self.y)

            for half_exponent in range(21):
                step_size = (1 / 2) ** half_exponent
                suggested_beta = step_size * beta_trial + (1 - step_size) * beta_previous
                objective_suggested = objective_function_value(X, D, suggested_beta, self.y)

                if objective_suggested <= objective_previous:
                    log.info(f"Using step length: {step_size}")
                    beta = suggested_beta
                    break
            else:
                log.info("Step halving routine failed...")
                break

            # beta = step_size * beta_suggestion + betas[-1] * (1 - step_size)
            log.info(f"Beta estimate with norm: {sp.linalg.norm(beta)}")
            log.info(f"Objective function value: {objective_function_value(X, D, beta, self.y)}")

            eta = X @ beta
            mu = self.link.inverse_link(eta)
            betas.append(beta)
            deviances.append(self.distribution.deviance(y=self.y, mu=mu).mean())

            if self._should_stop(betas=betas, step_size=step_size):
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

    def _should_stop(self, *, betas, step_size):
        if len(betas) < 2:
            return False

        # Stopping criteria from
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet

        diffs = betas[-1] - betas[-2]
        assert np.all(np.isfinite(diffs))
        max_coord_update = np.max(np.abs(diffs))
        max_coord = np.max(np.abs(betas[-1]))
        log.info(f"Stopping criteria evaluation: {max_coord_update:.6f} <= {self.tol} * {max_coord:.6f} ")
        return max_coord_update / max_coord < self.tol * step_size


class BetaOptimizer(Optimizer):
    """The most straightforward and simple way to fit a GAM,
    ignoring almost all concerns about speed and numercial stability."""

    step_size = 0.5

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
        threshold = EPSILON**0.25
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

        # Set non-identifiable coefficients to zero
        # Compute Q R = A P
        # https://en.wikipedia.org/wiki/QR_decomposition#Column_pivoting
        Q, R, P = sp.linalg.qr(np.vstack((self.X, self.D)), mode="economic", pivoting=True)
        zero_coefs = (np.abs(np.diag(R)) < EPSILON**0.5)[P]
        log.info(f"Number of non-identifiable coefficients set to zero: {zero_coefs.sum()}")
        X = self.X[:, ~zero_coefs]
        XT = X.T
        D = self.D[:, ~zero_coefs]
        DT = D.T
        betas[0] = betas[0][~zero_coefs]

        for iteration in range(1, self.max_iter + 1):
            log.info(f"Iteration {iteration}")

            # Step 1: Compute pseudodata z and iterative weights w
            # -----------------------------------------------------------------
            z = self.link.derivative(mu) * (self.y - mu) / alpha + eta
            w = alpha / (self.link.derivative(mu) ** 2 * self.distribution.V(mu))
            assert np.all(w > 0)

            # Step 3: Find beta to solve the weighted least squares objective
            # Solve f(z) = |z - X beta|^2_W + |D beta|^2
            # -----------------------------------------------------------------

            # Compute Q R = sqrt(W) X
            Q, R, pivot = sp.linalg.qr(np.sqrt(w).reshape(-1, 1) * X, mode="economic", pivoting=True)

            # Pirvot back
            pivot_inv = np.zeros_like(pivot)
            pivot_inv[pivot] = np.arange(len(pivot))
            R = R[pivot_inv, :]

            # Compute U @ D @ VT = vstack(R, D)
            U, d, VT = sp.linalg.svd(np.vstack([R, D[:, pivot]]), full_matrices=False)
            d_inv = 1 / d
            print(d_inv)
            assert np.all(d_inv > 0)

            d_inv[d_inv > 1000] = 0
            d_invVT = d_inv.reshape(1, -1) * VT

            beta_suggestion = np.linalg.multi_dot((d_invVT.T, d_invVT, XT[pivot, :], w * z))
            beta_suggestion = beta_suggestion[pivot_inv]

            # =============================================================================
            #             lhs = np.vstack([np.sqrt(w).reshape(-1, 1) * X, D])
            #             rhs = np.zeros(lhs.shape[0])
            #             rhs[:len(w)] = np.sqrt(w) * z
            #             beta_suggestion, *_ = sp.linalg.lstsq(lhs, rhs)
            #
            # =============================================================================

            self.step_size = 0.05

            beta = betas[-1] * (1 - self.step_size) + beta_suggestion * self.step_size

            # beta = step_size * beta_suggestion + betas[-1] * (1 - step_size)
            log.info(f"Beta estimate with norm: {sp.linalg.norm(beta)}")

            eta = X @ beta
            mu = self.link.inverse_link(eta)
            betas.append(beta)
            deviances.append(self.distribution.deviance(y=self.y, mu=mu).mean())

            if self._should_stop(betas):
                log.warning(f"Solved converged after {iteration} iterations.")
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
    import matplotlib.pyplot as plt
    from generalized_additive_models.links import Logit

    from sklearn.datasets import load_breast_cancer

    rng = np.random.default_rng(3)

    # Create a logistic problem
    x = np.linspace(0, 2 * np.pi, num=100000)
    X = x.reshape(-1, 1)
    linear_prediction = 1 + np.sin(x)

    X = rng.normal(size=(10000, 2))
    linear_prediction = 1 + np.sin(X[:, 0]) + np.cos(X[:, 1])

    mu = Logit().inverse_link(linear_prediction)

    y = rng.binomial(n=1, p=mu)

    # Create a GAM
    gam = GAM(Spline(None, extrapolation="periodic"), link="logit", distribution=Binomial(trials=1), max_iter=1000).fit(
        X, y
    )

    for term in gam.terms:
        if not isinstance(term, (Linear, Spline)):
            continue

        results = gam.partial_effect(term)

        plt.figure(figsize=(8, 3))
        plt.title(term.feature)
        plt.plot(results.x, results.y, color="red", zorder=10)
        plt.fill_between(results.x, results.y_low, results.y_high, alpha=0.5)
        # plt.scatter(results.x_obs, np.zeros_like(results.x_obs), marker="|", color="black")

        plt.scatter(results.x_obs, results.y_partial_residuals, color="black", s=1, alpha=1)

        # plt.plot(results.x, results.simulations.T, color="k", alpha=0.1)
        plt.ylim([-3, 3])

        plt.show()
