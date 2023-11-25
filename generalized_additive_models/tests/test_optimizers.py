#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:29:23 2023

@author: tommy
"""

import numpy as np
import pytest
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from generalized_additive_models import GAM, Linear


class TestOptimizationMethodsAgainstSklearn:
    # solver

    @pytest.mark.parametrize("num_features", [3, 5, 10])
    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("solver", (GAM._parameter_constraints["solver"][0]).options)
    def test_that_optimizers_produce_equal_results_on_Poisson_problem(self, solver, seed, num_features):
        """Check all GAM solvers against sklearn, implicitly testing them against each other."""

        # Dimensions
        rng = np.random.default_rng(seed)
        num_samples = 100

        # Create Poisson problem
        X = rng.standard_normal(size=(num_samples, num_features))
        beta = np.linspace(0.1, 1, num=num_features)
        linear_prediction = X @ beta
        mu = np.exp(linear_prediction)
        y = rng.poisson(lam=mu)
        sample_weight = np.exp(rng.standard_normal(size=y.shape))

        # Solve GLM using sklearn
        poisson_sklearn = PoissonRegressor(
            alpha=0,
            fit_intercept=False,
        ).fit(X, y, sample_weight=sample_weight)

        # Create a GLM
        terms = sum(Linear(i, penalty=0) for i in range(num_features))
        poisson_gam = GAM(
            terms,
            link="log",
            distribution="poisson",
            fit_intercept=False,
            solver=solver,  # <- Solver here :)
        ).fit(X, y, sample_weight=sample_weight)

        # Check that coefficients are similar
        assert np.allclose(poisson_sklearn.coef_, poisson_gam.coef_, atol=1e-3)

        # Deviances using sklearn objective
        dev_sklearn = mean_poisson_deviance(y_true=y, y_pred=poisson_sklearn.predict(X), sample_weight=sample_weight)
        dev_gam = mean_poisson_deviance(y_true=y, y_pred=poisson_gam.predict(X), sample_weight=sample_weight)

        # We want low deviance, so ideally we would have < 1 here.
        assert dev_sklearn / dev_gam < 1 + 1e-7


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v", "--capture=sys"])
