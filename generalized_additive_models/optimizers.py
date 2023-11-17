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

from generalized_additive_models.utils import phi_fletcher, ColumnRemover

MACHINE_EPSILON = np.finfo(float).eps
EPSILON = np.sqrt(MACHINE_EPSILON)


class Optimizer:
    def _validate_params(self):
        low, high = self.link.domain
        if np.any(self.y > high):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but largest y was: {self.y.max()}")

        if np.any(self.y < low):
            raise ValueError(f"Domain of {self.link} is {self.link.domain}, but smallest y was: {self.y.min()}")

    def _validate_outputs(self):
        results_keys = ("", "", "", "")
        assert all((key in self._statistics.keys()) for key in results_keys)

    def evaluate_objective(self, beta, *, X, D, y, sample_weight):
        """Evaluate the log likelihood plus the penalty."""
        mu = self.link.inverse_link(X @ beta)
        deviance = self.distribution.deviance(y=y, mu=mu, scaled=True, sample_weight=sample_weight).sum()
        penalty = sp.linalg.norm(D @ beta) ** 2

        # print("-------------start --------------")
        # print(self.distribution.deviance(y=y, mu=mu, scaled=True, sample_weight=sample_weight).mean())
        # from sklearn.metrics import mean_poisson_deviance
        # print(mean_poisson_deviance(y_true=y, y_pred=mu, sample_weight=sample_weight))
        # print("-------------end --------------")

        return (deviance + penalty) / len(beta)

    def initial_estimate(self):
        """Construct an initial estimate of beta by solving a Ridge problem.

        The idea is to take the observations y, map them to the unbounded linear
        space by mu = g(y), then solve the equation X @ beta = mu.
        In summary, we use Ridge to solve:

            X @ beta = g(y)
        """
        # TODO: even in the initial estimate, D could be used

        # Numerical problems occur with e.g. logit link, since 0 and 1 map to inf
        low, high = self.link.domain
        threshold = EPSILON**0.25
        y_to_map = np.maximum(np.minimum(self.y, high - threshold), low + threshold)
        mu_initial = self.link.link(y_to_map)

        assert np.all(np.isfinite(mu_initial)), "Initial `mu` must be finite."

        # Solve X @ beta = g(y) = mu using Ridge regression
        ridge = Ridge(alpha=1e3, fit_intercept=False)
        sample_weight = self.get_sample_weight(mu=mu_initial, y=self.y)
        ridge.fit(self.X, mu_initial, sample_weight=sample_weight)

        # Respect the bounds naively by projecting to them
        lower_bound, upper_bound = self.bounds
        return np.maximum(np.minimum(ridge.coef_, upper_bound), lower_bound)


class NelderMead(Optimizer):
    def __init__(
        self,
        *,
        X,
        D,
        y,
        link,
        distribution,
        bounds,
        max_iter,
        tol,
        get_sample_weight,
        verbose,
    ):
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

    def solve(self):
        beta = self.initial_estimate()
        eta = self.X @ beta
        mu = self.link.inverse_link(eta)

        # Bind function arguments
        objective_function = functools.partial(
            self.evaluate_objective, X=self.X, D=self.D, y=self.y, sample_weight=self.get_sample_weight(mu=mu, y=self.y)
        )

        result = sp.optimize.minimize(
            objective_function,
            x0=beta,
            method="nelder-mead",
            bounds=list(zip(*self.bounds)),
            tol=self.tol,
            callback=None,
            options={"maxiter": self.max_iter * 10},
        )

        return result.x


class PIRLS(Optimizer):
    """The most straightforward and simple way to fit a GAM,
    ignoring almost all concerns about speed and numerical stability."""

    # Printing options
    PRECISION = 4
    MIN_DIGITS = 4
    EXP_DIGITS = 2

    def __init__(
        self,
        *,
        X,
        D,
        y,
        link,
        distribution,
        bounds,
        max_iter,
        tol,
        get_sample_weight,
        verbose,
    ):
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

        self._validate_params()

    def solve_unbounded_lstsq(self, *, X, D, w, z):
        """Solve (X.T @ diag(w) @ X + D.T @ D) beta = X.T @ diag(w) @ z for beta."""
        lhs = X.T @ (w.reshape(-1, 1) * X) + D.T @ D
        rhs = X.T @ (w * z)
        beta, *_ = sp.linalg.lstsq(
            lhs, rhs, cond=None, overwrite_a=True, overwrite_b=True, check_finite=True, lapack_driver="gelsd"
        )
        return beta

    def solve_lstsq(self, X, D, w, z, bounds):
        """Solve (X.T @ diag(w) @ X + D.T @ D) beta = X.T @ diag(w) @ z"""

        lower_bounds, upper_bounds = bounds

        # If bounds are inactive, solve using standard least squares
        if np.all(lower_bounds == -np.inf) and np.all(upper_bounds == np.inf):
            return self.solve_unbounded_lstsq(X=X, D=D, w=w, z=z)

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

    def halving_search(self, X, D, y, sample_weight, beta0, beta1):
        """Perform halving search.

        Here beta0 is the solution to the previous Newton step, and beta1 is
        the proposed solution by the current step. In practice beta1 sometimes
        overshoots the optimum, so we succesively halv the step until we improve
        on beta0.

        iter1 -------------------------------->
        iter2 ---------------->
        iter3 -------->
        iter4 ---->
        ___________________________________________
        beta0                                beta1

        - The box constraints on the variables are respected by the solution if
          both beta0 and beta1 respect the constraints, since it's a convex
          combination of two points within a high-dimensional box [min, max]^D.
        - I also tested using sp.optimize.minimize_scalar, but found that using
          halving search is equally good as easier.
        """
        # Starting objective value
        obj0 = self.evaluate_objective(beta=beta0, X=X, D=D, y=y, sample_weight=sample_weight)

        # Try step sizes 1, 1/2, 1/4, 1/8, ..., 1/2^19 = 1.9e-06
        for iteration in range(20):
            step_size = (1 / 2) ** iteration
            beta = step_size * beta1 + (1 - step_size) * beta0
            obj = self.evaluate_objective(beta=beta, X=X, D=D, y=y, sample_weight=sample_weight)

            # Found a better solution, return it
            if obj < obj0:
                return beta, obj, iteration

        # No better solution found
        return beta0, obj0, iteration

    def alpha(self, mu):
        # Fisher weights, see page 250 in Wood, 2nd ed
        if self.fisher_weights:
            alpha = 1
        else:
            V_term = self.distribution.V_derivative(mu) / self.distribution.V(mu)
            g_term = self.link.second_derivative(mu) / self.link.derivative(mu)
            alpha = 1 + (self.y - mu) * (V_term + g_term)

        return alpha

    def solve(self, fisher_weights=True):
        """Solve the optimization problem."""
        # Page 106 in Wood, 2nd ed: 3.1.2 Fitting generalized linear models
        self.fisher_weights = fisher_weights

        num_observations, num_beta = self.X.shape

        # Number formatting when printing
        fmt = functools.partial(
            np.format_float_scientific, precision=self.PRECISION, min_digits=self.MIN_DIGITS, exp_digits=self.EXP_DIGITS
        )

        # See page 251 in Wood, 2nd edition
        # Step 1: Compute initial values
        # ---------------------------------------------------------------------
        beta = self.initial_estimate()

        eta = self.X @ beta
        mu = self.link.inverse_link(eta)
        sample_weight = self.get_sample_weight(mu=mu, y=self.y)

        if self.verbose >= 1:
            lpad = int(np.floor(np.log10(self.max_iter)))
            objective_init = self.evaluate_objective(
                beta=beta, X=self.X, D=self.D, y=self.y, sample_weight=sample_weight
            )

            msg = f"Initial guess:      Objective: {fmt(objective_init)}   "
            beta_fmt = fmt(np.sqrt(np.mean(beta**2)))
            msg += f"Coef. rmse: {beta_fmt}   "
            print(msg)

        # Set non-identifiable coefficients to zero
        column_remover = ColumnRemover()
        X, D, beta = column_remover.transform(X=self.X, D=self.D, beta=beta)

        if self.verbose >= 2:
            print(
                "Variables set to zero for identifiability:"
                + f"{column_remover.zero_coefs.sum()}/{len(column_remover.zero_coefs)}"
            )

        # List of betas to watch as optimization progresses
        betas = [beta]
        deviances = [self.distribution.deviance(y=self.y, mu=mu, sample_weight=sample_weight).mean()]

        for iteration in range(1, self.max_iter + 1):
            # Step 1: Compute pseudodata z and iterative weights w
            z = self.link.derivative(mu) * (self.y - mu) / self.alpha(mu) + eta

            sample_weight = self.get_sample_weight(mu=mu, y=self.y)
            w = sample_weight * self.alpha(mu) / (self.link.derivative(mu) ** 2 * self.distribution.V(mu))
            assert np.all(w >= 0), f"smallest w_i was negative: {np.min(w)}"

            # Step 3: Find beta to solve the weighted least squares objective
            # Solve f(z) = |z - X @ beta|^2_W + |D @ beta|^2
            bounds = (self.bounds[0][column_remover.nonzero_coefs], self.bounds[1][column_remover.nonzero_coefs])
            beta_trial = self.solve_lstsq(X, D, w, z, bounds=bounds)

            beta, objective_value, half_exponent = self.halving_search(
                X, D, self.y, sample_weight, betas[-1], beta_trial
            )

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
        betas = [column_remover.insert(initial=np.zeros(num_beta), values=beta) for beta in betas]
        beta = betas[-1]

        # Compute the hat matrix: H = X @ (X.T @ W @ X + D.T @ D)^-1 @ W @ X.T
        # Also called the projection matrix or influence matrix (page 251 in Wood, 2nd ed)
        to_invert = X.T @ (w.reshape(-1, 1) * X) + D.T @ D
        np.fill_diagonal(to_invert, to_invert.diagonal() + EPSILON)  # Add to diagonal
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
        H_diag = column_remover.insert(initial=np.zeros_like(beta), values=H_diag)

        assert len(beta) == len(H_diag)

        edof_per_coef = H_diag
        edof = edof_per_coef.sum()

        # Compute phi by equation (6.2) (page 251 in Wood, 2nd ed; also p 110)
        # In the Gaussian case, phi is the variance
        if self.distribution.scale is not None:
            phi = self.distribution.scale
        else:
            phi = phi_fletcher(
                self.y,
                mu,
                self.distribution,
                edof,
                sample_weight=self.get_sample_weight(mu=mu, y=self.y),
            )

        self.results_.scale = phi

        # Compute the covariance matrix of the parameters V_\beta (page 293 in Wood, 2nd ed)
        covariance = inverted * phi
        covariance = column_remover.insert(initial=np.eye(len(beta), dtype=float) * EPSILON, values=covariance)

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

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules"])

    from generalized_additive_models import Linear, GAM

    rng = np.random.default_rng(42)
    num_features = 1
    num_samples = 10

    # Create a poisson problem
    X = rng.standard_normal(size=(num_samples, num_features))
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler(with_std=False).fit_transform(X)
    beta = np.arange(num_features) + 1
    linear_prediction = X @ beta
    mu = np.exp(linear_prediction) + 0
    y = rng.poisson(lam=mu)
    sample_weight = np.ones_like(y, dtype=float)
    sample_weight[-25:] = 100
    sample_weight = np.ones_like(y, dtype=float)

    # Sklearn model
    from sklearn.linear_model import PoissonRegressor

    poisson_sklearn = PoissonRegressor(
        alpha=0,
        fit_intercept=False,
    ).fit(X, y, sample_weight=sample_weight)

    # Create a GAM
    terms = sum(Linear(i, penalty=0) for i in range(num_features))
    poisson_gam = GAM(
        terms,
        link="log",
        distribution="poisson",
        fit_intercept=False,
        verbose=10,
        # tol=1e-200,
    ).fit(X, y, sample_weight=sample_weight)

    print(poisson_gam.coef_)

    print("-----------------------------------------------------------")

    optimizer = NelderMead(
        X=poisson_gam.model_matrix_,
        D=poisson_gam.terms.penalty_matrix(),
        y=y,
        link=poisson_gam._link,
        distribution=poisson_gam._distribution,
        bounds=(poisson_gam.terms._lower_bound, poisson_gam.terms._upper_bound),
        get_sample_weight=functools.partial(poisson_gam._get_sample_weight, sample_weight=sample_weight),
        max_iter=poisson_gam.max_iter,
        tol=poisson_gam.tol,
        verbose=poisson_gam.verbose,
    )

    beta_mead = optimizer.solve()

    print(beta_mead)

    print("-----------------------------------------------------------")

    optimizer = PIRLS(
        X=poisson_gam.model_matrix_,
        D=poisson_gam.terms.penalty_matrix(),
        y=y,
        link=poisson_gam._link,
        distribution=poisson_gam._distribution,
        bounds=(poisson_gam.terms._lower_bound, poisson_gam.terms._upper_bound),
        get_sample_weight=functools.partial(poisson_gam._get_sample_weight, sample_weight=sample_weight),
        max_iter=poisson_gam.max_iter,
        tol=poisson_gam.tol,
        verbose=poisson_gam.verbose,
    )

    beta_pirls = optimizer.solve()

    print("-----------------------------------------------------------")
    print(f"mead {beta_mead}")
    print(f"pirls {beta_pirls}")
    print(f"sklearn {poisson_sklearn.coef_}")

    obj_mead = optimizer.evaluate_objective(
        beta=beta_mead,
        X=poisson_gam.model_matrix_,
        D=poisson_gam.terms.penalty_matrix(),
        y=y,
        sample_weight=sample_weight,
    ).round(6)

    obj_pirls = optimizer.evaluate_objective(
        beta=beta_pirls,
        X=poisson_gam.model_matrix_,
        D=poisson_gam.terms.penalty_matrix(),
        y=y,
        sample_weight=sample_weight,
    ).round(6)

    obj_sklearn = optimizer.evaluate_objective(
        beta=poisson_sklearn.coef_,
        X=X,
        D=poisson_gam.terms.penalty_matrix(),
        y=y,
        sample_weight=sample_weight,
    ).round(6)

    print(f"gam objective mead: {obj_mead:,}")
    print(f"gam objective pirls: {obj_pirls:,}")
    print(f"gam objective sklearn: {obj_sklearn:,}")

    from sklearn.metrics import mean_poisson_deviance

    print("--------------- sklearn metrics ------------")

    obj_mead = mean_poisson_deviance(
        y_true=y, y_pred=poisson_gam._link.inverse_link(poisson_gam.model_matrix_ @ beta_mead)
    )
    obj_pirls = mean_poisson_deviance(y_true=y, y_pred=poisson_gam.predict(X))
    obj_sklearn = mean_poisson_deviance(y_true=y, y_pred=poisson_sklearn.predict(X))

    print(f"sklearn objective mead: {obj_mead:,}")
    print(f"sklearn objective pirls: {obj_pirls:,}")
    print(f"sklearn objective sklearn: {obj_sklearn:,}")

    print("------------------------------------------")
