#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 06:22:30 2023

@author: tommy
"""
import warnings
import functools

import numpy as np
import scipy as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.utils import Bunch

from generalized_additive_models.utils import identifiable_parameters, phi_fletcher

MACHINE_EPSILON = np.finfo(float).eps
EPSILON = np.sqrt(MACHINE_EPSILON)


def solve_unbounded_lstsq(X, D, w, z):
    """Solve (X.T @ diag(w) @ X + D.T @ D) beta = X.T @ diag(w) @ z
    for beta."""
    lhs = X.T @ (w.reshape(-1, 1) * X) + D.T @ D
    rhs = X.T @ (w * z)
    beta, *_ = sp.linalg.lstsq(lhs, rhs, cond=None, overwrite_a=True, overwrite_b=True)
    return beta


class Optimizer:
    def _validate_params(self):
        pass

    def _validate_outputs(self):
        results_keys = ("", "", "", "")
        assert all((key in self._statistics.keys()) for key in results_keys)


class PIRLS(Optimizer):
    """The most straightforward and simple way to fit a GAM,
    ignoring almost all concerns about speed and numercial stability."""

    def __init__(self, *, X, D, y, link, distribution, bounds, max_iter, tol, get_sample_weight, verbose):
        self.X = X
        self.D = D
        self.y = y
        self.link = link
        self.distribution = distribution
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol
        self.results_ = Bunch()
        self.get_sample_weight = get_sample_weight
        self.verbose = verbose

    def _validate_params(self):
        low, high = self.link.domain
        if np.any(self.y < high):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but largest y was: {self.y.max()}")

        if np.any(self.y > high):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but largest y was: {self.y.max()}")

    def solve_lstsq(self, X, D, w, z, bounds):
        """Solve (X.T @ diag(w) @ X + D.T @ D) beta = X.T @ diag(w) @ z"""

        lower_bounds, upper_bounds = bounds

        # If bounds are inactive, solve using standard least squares
        if np.all(lower_bounds == -np.inf) and np.all(upper_bounds == np.inf):
            return solve_unbounded_lstsq(X, D, w, z)

        # Set up left hand side and right hand side
        lhs = X.T @ (w.reshape(-1, 1) * X) + D.T @ D
        rhs = X.T @ (w * z)

        verbose = min(max(0, self.verbose - 2), 2)
        result = sp.optimize.lsq_linear(lhs, rhs, bounds=bounds, verbose=verbose)

        if self.verbose >= 2:
            msg = f"  Constrained LSQ {'success' if result.success else 'failure'} "
            msg += f"in {result.nit} iters. Msg: {result.message}"
            print(msg)

            if not result.success:
                print(f" => Constrained LSQ msg: {result.message}")

        return result.x

    def _initial_estimate(self):
        """Construct an initial estimate of beta by solving a Ridge problem."""

        # Numerical problems occur with e.g. logit link, since 0 and 1 map to inf
        low, high = self.link.domain
        threshold = EPSILON**0.25
        y_to_map = np.maximum(np.minimum(self.y, high - threshold), low + threshold)
        mu_initial = self.link.link(y_to_map)

        assert np.all(np.isfinite(mu_initial)), "Initial `mu` must be finite."

        # Solve X @ beta = mu using Ridge regression
        ridge = Ridge(alpha=1e3, fit_intercept=False)
        sample_weight = self.get_sample_weight(mu=mu_initial, y=self.y)
        ridge.fit(self.X, mu_initial, sample_weight=sample_weight)

        # Respect the bounds naively by projecting to them
        lower_bound, upper_bound = self.bounds
        return np.maximum(np.minimum(ridge.coef_, upper_bound), lower_bound)

    def evaluate_objective(self, X, D, beta, y, sample_weight):
        """Evaluate the log likelihood plus the penalty."""
        mu = self.link.inverse_link(X @ beta)
        deviance = self.distribution.deviance(y=y, mu=mu, scaled=True, sample_weight=sample_weight).sum()
        penalty = sp.linalg.norm(D @ beta) ** 2
        return (deviance + penalty) / len(beta)

    def solve(self, fisher_weights=True):
        """Solve the optimization problem."""
        # Page 106 in Wood, 2nd ed: 3.1.2 Fitting generalized linear models

        num_observations, num_beta = self.X.shape

        def alpha(mu):
            # Fisher weights, see page 250 in Wood, 2nd ed
            if fisher_weights:
                alpha = 1
            else:
                V_term = self.distribution.V_derivative(mu) / self.distribution.V(mu)
                g_term = self.link.second_derivative(mu) / self.link.derivative(mu)
                alpha = 1 + (self.y - mu) * (V_term + g_term)

            return alpha

        fmt = functools.partial(np.format_float_scientific, precision=4, min_digits=4, exp_digits=2)

        # See page 251 in Wood, 2nd edition
        # Step 1: Compute initial values
        # ---------------------------------------------------------------------
        beta = self._initial_estimate()

        eta = self.X @ beta
        mu = self.link.inverse_link(eta)
        sample_weight = self.get_sample_weight(mu=mu, y=self.y)

        if self.verbose >= 1:
            lpad = int(np.floor(np.log10(self.max_iter)))
            w = alpha(mu) * sample_weight / (self.link.derivative(mu) ** 2 * self.distribution.V(mu))
            objective_init = self.evaluate_objective(self.X, self.D, beta, self.y, w)

            msg = f"Initial guess:      Objective: {fmt(objective_init)}   "
            beta_fmt = fmt(np.sqrt(np.mean(beta**2)))
            msg += f"Coef. rmse: {beta_fmt}   "
            print(msg)

        # Set non-identifiable coefficients to zero
        nonzero_coefs = identifiable_parameters(np.vstack((self.X, self.D)))
        zero_coefs = ~nonzero_coefs

        if self.verbose >= 2:
            print(f"Variables set to zero for identifiability: {zero_coefs.sum()}/{len(zero_coefs)}")

        X = self.X[:, nonzero_coefs]
        D = self.D[:, nonzero_coefs]
        beta = beta[nonzero_coefs]

        # List of betas to watch as optimization progresses
        betas = [beta]
        deviances = [self.distribution.deviance(y=self.y, mu=mu, sample_weight=sample_weight).mean()]

        for iteration in range(1, self.max_iter + 1):
            # Step 1: Compute pseudodata z and iterative weights w
            z = self.link.derivative(mu) * (self.y - mu) / alpha(mu) + eta

            sample_weight = self.get_sample_weight(mu=mu, y=self.y)
            w = alpha(mu) * sample_weight / (self.link.derivative(mu) ** 2 * self.distribution.V(mu))
            assert np.all(w >= 0), f"smallest w_i {np.min(w)}"

            # Step 3: Find beta to solve the weighted least squares objective
            # Solve f(z) = |z - X @ beta|^2_W + |D @ beta|^2
            bounds = (self.bounds[0][nonzero_coefs], self.bounds[1][nonzero_coefs])
            beta_trial = self.solve_lstsq(X, D, w, z, bounds=bounds)

            def halving_search(X, D, y, sample_weight, beta0, beta1):
                obj0 = self.evaluate_objective(X, D, beta0, y, sample_weight)

                for half_exponent in range(21):
                    step_size = (1 / 2) ** half_exponent
                    beta = step_size * beta1 + (1 - step_size) * beta0
                    obj = self.evaluate_objective(X, D, beta, y, sample_weight)

                    if obj < obj0:
                        return beta, obj, half_exponent

                return beta0, obj0, half_exponent

            beta, objective_value, half_exponent = halving_search(X, D, self.y, sample_weight, betas[-1], beta_trial)

            eta = X @ beta
            mu = self.link.inverse_link(eta)
            betas.append(beta)
            deviances.append(self.distribution.deviance(y=self.y, mu=mu, sample_weight=sample_weight).mean())

            if self.verbose >= 1:
                lpad = int(np.floor(np.log10(self.max_iter)))

                msg = f"Iteration: {str(iteration).rjust(lpad, ' ')}/{self.max_iter}   "
                msg += f"Objective: {fmt(objective_value)}   "
                msg += f"Coef. rmse: {fmt(np.sqrt(np.mean(beta**2)))}   "
                msg += f"Step size: 1/2^{half_exponent}"
                print(msg)

            if self._should_stop(betas=betas, step_size=1):
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
        to_invert = X.T @ (w.reshape(-1, 1) * X) + D.T @ D
        to_invert.flat[:: to_invert.shape[0] + 1] += EPSILON  # Add to diagonal
        inverted = sp.linalg.inv(to_invert)

        # Compute the effective degrees of freedom per coefficient
        # Only need the diagonal of H, so use the fact that
        # np.diag(B @ A.T) = (B * A).sum(axis=1)
        # to compute H below:
        # H = ((w.reshape(-1, 1) * self.X) @ inverted @ self.X.T)
        # H_diag = np.sum(((w.reshape(-1, 1) * self.X) @ inverted) * self.X, axis=1)
        # H_diag2 = sp.linalg.inv(self.X.T @ (w.reshape(-1, 1) * self.X) + self.D.T @ self.D) @ self.X.T @ np.diag(w) @ self.X
        H_diag = ((X.T @ (w.reshape(-1, 1) * X)) * inverted).sum(axis=1)
        # H_diag = np.diag(inverted @ self.X.T @ np.diag(w) @ self.X )
        # assert np.allclose(H_diag, np.diag(H_diag2))

        # assert len(beta) == len(np.diag(H_diag2))
        H_diag_full = np.zeros(num_beta, dtype=float)
        H_diag_full[nonzero_coefs] = H_diag
        H_diag = H_diag_full

        assert len(beta) == len(H_diag)

        edof_per_coef = H_diag
        edof = edof_per_coef.sum()

        # Compute phi by equation (6.2) (page 251 in Wood, 2nd ed; also p 110)
        # In the Gaussian case, phi is the variance
        if self.distribution.scale is not None:
            phi = self.distribution.scale
        else:
            phi = phi_fletcher(
                self.y, mu, self.distribution, edof, sample_weight=self.get_sample_weight(mu=mu, y=self.y)
            )

        self.results_.scale = phi

        # Compute the covariance matrix of the parameters V_\beta (page 293 in Wood, 2nd ed)
        covariance = inverted * phi
        covariance_full = np.zeros(shape=(len(beta), len(beta)), dtype=float)
        covariance_full = np.eye(len(beta), dtype=float) * EPSILON
        covariance_full[np.ix_(nonzero_coefs, nonzero_coefs)] = covariance
        covariance = covariance_full

        assert covariance.shape == (len(beta), len(beta))
        self.results_.covariance = covariance

        self.results_.edof_per_coef = edof_per_coef
        self.results_.edof = edof
        self.results_.iters_deviance = deviances
        self.results_.iters_coef = betas

        # Compute generalized cross validation score
        # Equation (6.18) on page 260
        # TODO: This is only the Gaussian case, see page 262
        gcv = sp.linalg.norm(self.X @ beta - self.y) ** 2 * len(self.y) / (len(self.y) - edof) ** 2
        self.results_.generalized_cross_validation_score = gcv

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
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])
