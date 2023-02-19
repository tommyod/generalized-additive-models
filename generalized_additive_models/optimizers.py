#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 06:22:30 2023

@author: tommy
"""
import warnings

import numpy as np
import scipy as sp
from sklearn.exceptions import ConvergenceWarning
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


class PIRLS(Optimizer):
    """The most straightforward and simple way to fit a GAM,
    ignoring almost all concerns about speed and numercial stability."""

    def __init__(self, *, X, D, y, link, distribution, max_iter, tol, verbose):
        self.X = X
        self.D = D
        self.y = y
        self.link = link
        self.distribution = distribution
        self.max_iter = max_iter
        self.tol = tol
        self.statistics_ = Bunch()
        self.verbose = verbose

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
        return beta_initial

    def evaluate_objective(self, X, D, beta, y):
        """Evaluate the log likelihood plus the penalty."""
        mu = self.link.inverse_link(X @ beta)
        deviance = self.distribution.deviance(y=y, mu=mu, scaled=True).sum()
        penalty = sp.linalg.norm(D @ beta) ** 2
        return (deviance + penalty) / len(beta)

    def solve(self):
        num_observations, num_beta = self.X.shape

        beta = self._initial_estimate()

        # Step 1: Compute initial values
        eta = self.X @ beta
        mu = self.link.inverse_link(eta)

        if self.verbose >= 1:
            lpad = int(np.floor(np.log10(self.max_iter)))
            objective_init = self.evaluate_objective(self.X, self.D, beta, self.y)

            msg = "Initial guess:      "
            objective_fmt = np.format_float_scientific(objective_init, precision=4, min_digits=4)
            msg += f"Objective: {objective_fmt}   "
            beta_fmt = np.format_float_scientific(sp.linalg.norm(beta), precision=4, min_digits=4)
            msg += f"Coef. norm: {beta_fmt}   "
            print(msg)

        # List of betas to watch as optimization progresses
        betas = [beta]
        deviances = [self.distribution.deviance(y=self.y, mu=mu).mean()]

        alpha = 1  # Fisher weights, see page 250 in Wood, 2nd ed

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
            objective_previous = self.evaluate_objective(X, D, beta_previous, self.y)

            for half_exponent in range(21):
                step_size = (1 / 2) ** half_exponent
                suggested_beta = step_size * beta_trial + (1 - step_size) * beta_previous
                objective_suggested = self.evaluate_objective(X, D, suggested_beta, self.y)

                if objective_suggested <= objective_previous:
                    beta = suggested_beta
                    break
            else:
                break

            eta = X @ beta
            mu = self.link.inverse_link(eta)
            betas.append(beta)
            deviances.append(self.distribution.deviance(y=self.y, mu=mu).mean())

            if self.verbose >= 1:
                lpad = int(np.floor(np.log10(self.max_iter)))

                msg = f"Iteration: {str(iteration).rjust(lpad, ' ')}/{self.max_iter}   "
                objective_fmt = np.format_float_scientific(objective_suggested, precision=4, min_digits=4, exp_digits=2)
                msg += f"Objective: {objective_fmt}   "
                beta_fmt = np.format_float_scientific(sp.linalg.norm(beta), precision=4, min_digits=4, exp_digits=2)
                msg += f"Coef. norm: {beta_fmt}   "
                msg += f"Step size: 1/2^{half_exponent}"
                print(msg)

            if self._should_stop(betas=betas, step_size=step_size):
                if self.verbose >= 1:
                    print(" => SUCCESS: Solver converged (met tolerance criterion).")
                break
        else:
            if self.verbose >= 1:
                print(f" => FAILURE: Solver did not converge in {self.max_iter} iterations.")

            msg = f"Solver did not converge in {self.max_iter} iterations.\n"
            msg += "Increase `max_iter`, increase `tol` or increase penalties."
            warnings.warn(msg, ConvergenceWarning)

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
        # log.info(f"Stopping criteria evaluation: {max_coord_update:.6f} <= {self.tol} * {max_coord:.6f} ")
        return max_coord_update / max_coord < self.tol * step_size


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import fetch_california_housing, load_breast_cancer
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from generalized_additive_models.distributions import Binomial
    from generalized_additive_models.gam import GAM
    from generalized_additive_models.links import Logit
    from generalized_additive_models.terms import Intercept, Linear, Spline, TermList

    rng = np.random.default_rng(3)

    # Create a logistic problem
    x = np.linspace(0, 2 * np.pi, num=10000)
    X = x.reshape(-1, 1)
    linear_prediction = 1 + np.sin(x)

    X = rng.normal(size=(1000, 4))
    linear_prediction = 1 + np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.cos(X[:, 2] * 2) + np.sin(X[:, 3] * 4)

    mu = Logit().inverse_link(linear_prediction)

    y = rng.binomial(n=1, p=mu)

    # Create a GAM
    gam = GAM(Spline(None, extrapolation="periodic"), link="logit", distribution=Binomial(trials=1), max_iter=100).fit(
        X, y
    )

    for term in gam.terms:
        if not isinstance(term, (Linear, Spline)):
            continue

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
