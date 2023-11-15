#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 08:25:44 2023

@author: tommy
"""


import numpy as np
import pandas as pd
import scipy as sp
import pytest

from sklearn.linear_model import Ridge, PoissonRegressor

from generalized_additive_models.gam import GAM
from generalized_additive_models.terms import Categorical, Linear
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_poisson_deviance


class TestAgainstRLM:
    def test_that_inference_of_factors_equals_Rs_lm_function(self):
        """

        height <- c(2.5, 1, 1.5, 1, 1, 0, 1, 0)
        gender <- factor(c("male","male","male","male","female","female","female", "female"))
        country <- factor(c("no", "us", "no", "us", "no", "us", "no", "us"))

        data <- data.frame(height, gender, country)

        model = lm(height ~ gender + country - 1, data=data)
        summary(model)

        ----------------------------
        Call:
        lm(formula = height ~ gender + country - 1, data = data)

        Residuals:
                 1          2          3          4          5          6          7          8
         5.000e-01 -1.943e-16 -5.000e-01  1.156e-16 -5.551e-17  2.776e-17 -5.551e-17  2.776e-17

        Coefficients:
                     Estimate Std. Error t value Pr(>|t|)
        genderfemale   1.0000     0.1936   5.164 0.003573 **
        gendermale     2.0000     0.1936  10.328 0.000146 ***
        countryus     -1.0000     0.2236  -4.472 0.006566 **
        ---
        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

        Residual standard error: 0.3162 on 5 degrees of freedom
        Multiple R-squared:   0.96,	Adjusted R-squared:  0.936
        F-statistic:    40 on 3 and 5 DF,  p-value: 0.0006425

        """

        df = pd.DataFrame(
            {
                "height": (2.5, 1, 1.5, 1, 1, 0, 1, 0),
                "gender": (
                    "male",
                    "male",
                    "male",
                    "male",
                    "female",
                    "female",
                    "female",
                    "female",
                ),
                "country": ("no", "us", "no", "us", "no", "us", "no", "us"),
            }
        )

        gender_cat = Categorical("gender", penalty=0)
        country_cat = Categorical("country", penalty=0)

        gam = GAM(gender_cat + country_cat, fit_intercept=False).fit(df, df.height)

        # Standard errors are bounded
        assert np.all(np.sqrt(np.diag(gender_cat.coef_covar_)) < 0.23)
        assert np.all(np.sqrt(np.diag(country_cat.coef_covar_)) < 0.23)

        # Residuals match R results
        residuals = (df.height - gam.predict(df)).values
        residuals_R = np.array(
            [
                5.000e-01,
                -1.943e-16,
                -5.000e-01,
                1.156e-16,
                -5.551e-17,
                2.776e-17,
                -5.551e-17,
                2.776e-17,
            ]
        )
        assert np.allclose(residuals, residuals_R)


class TestAgainstSklearnRidge:
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("penalty", np.logspace(-5, 5, num=11))
    def test_against_ridge_without_intercept(self, seed, penalty):
        # Create dataset
        rng = np.random.default_rng(seed)

        # The GAM will center each feature to have mean zero.
        # To have exactly the same effect of regularization, we must do the same here
        X = StandardScaler(with_std=False).fit_transform(rng.standard_normal(size=(99, 2)))
        beta = np.arange(X.shape[1]) + 1
        y = X @ beta

        # Solve the Ridge problem directly, using the fact that
        # |X @ beta - y|_2^2 + alpha * |beta|_2^2 =
        # |[X | sqrt(alpha)]^T @ beta - [y | 0]^T|_2^2
        lhs = np.vstack((X, np.sqrt(penalty) * np.eye(len(beta))))
        rhs = np.hstack((y, np.zeros(len(beta))))
        coefs_direct, *_ = sp.linalg.lstsq(lhs, rhs)

        # Perform regression with sklearn
        ridge = Ridge(alpha=penalty, solver="auto", fit_intercept=False).fit(X, y)

        # Perform regression with GAM
        gam = GAM(Linear(0, penalty=penalty) + Linear(1, penalty=penalty), fit_intercept=False).fit(X, y)

        # Check all against each other
        assert np.allclose(coefs_direct, ridge.coef_)
        assert np.allclose(gam.coef_, ridge.coef_)

    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("penalty", np.logspace(-5, 5, num=11))
    def test_against_ridge_with_intercept(self, seed, penalty):
        # Create dataset
        rng = np.random.default_rng(seed)

        # The GAM will center each feature to have mean zero.
        # To have exactly the same effect of regularization, we must do the same here
        X = StandardScaler(with_std=True).fit_transform(rng.standard_normal(size=(99, 2)))
        beta = np.arange(X.shape[1]) + 1
        y = X @ beta + 10

        # Solve the Ridge problem directly, using the fact that
        # |X @ beta - y|_2^2 + alpha * |beta|_2^2 =
        # |[X | sqrt(alpha)]^T @ beta - [y | 0]^T|_2^2
        X_with_intercept = np.hstack((X, np.ones(X.shape[0])[:, np.newaxis]))
        I_beta = np.eye(len(beta) + 1)
        I_beta[-1, -1] = 0
        lhs = np.vstack((X_with_intercept, np.sqrt(penalty) * I_beta))
        rhs = np.hstack((y, np.zeros(len(beta) + 1)))
        coefs_direct, *_ = sp.linalg.lstsq(lhs, rhs)

        # Perform regression with sklearn
        ridge = Ridge(alpha=penalty, solver="auto", fit_intercept=True).fit(X, y)
        coefs_ridge = np.hstack((ridge.coef_, [ridge.intercept_]))

        # Perform regression with GAM
        gam = GAM(Linear(0, penalty=penalty) + Linear(1, penalty=penalty), fit_intercept=True).fit(X, y)

        # Check all against each other
        assert np.allclose(coefs_direct, coefs_ridge)
        assert np.allclose(gam.coef_, coefs_ridge)

    @pytest.mark.parametrize("seed", list(range(25)))
    @pytest.mark.parametrize("num_samples", [100, 1000])
    def test_against_poisson(self, seed, num_samples):
        rng = np.random.default_rng(seed)

        # Create a poisson problem
        X = rng.standard_normal(size=(num_samples, 2))
        beta = np.arange(2) + 1
        linear_prediction = X @ beta
        mu = np.exp(linear_prediction)
        y = rng.poisson(lam=mu)

        # Create scikit-learn model
        poisson_sklearn = PoissonRegressor(
            alpha=0,
            fit_intercept=False,
        ).fit(X, y)

        # Create a GAM
        poisson_gam = GAM(
            Linear(0, penalty=0) + Linear(1, penalty=0),
            link="log",
            distribution="poisson",
            fit_intercept=False,
            verbose=10,
            tol=1e-99,
        ).fit(X, y)

        # Compare deviance
        dev_sklearn = mean_poisson_deviance(y_true=y, y_pred=poisson_sklearn.predict(X))
        dev_gam = mean_poisson_deviance(y_true=y, y_pred=poisson_gam.predict(X))
        assert dev_gam / dev_sklearn < 1.5


if __name__ == "__main__":
    pytest.main(
        args=[
            __file__,
            "-v",
            "--capture=sys",
            "--doctest-modules",
        ]
    )

    rng = np.random.default_rng(42)

    # Create a poisson problem
    X = rng.standard_normal(size=(999, 2))
    beta = np.arange(2) + 1
    linear_prediction = X @ beta
    mu = np.exp(linear_prediction)
    y = rng.poisson(lam=mu)

    # Create scikit-learn model
    poisson_sklearn = PoissonRegressor(
        alpha=0,
        fit_intercept=False,
    ).fit(X, y)

    # Create a GAM
    poisson_gam = GAM(
        Linear(0, penalty=0) + Linear(1, penalty=0),
        link="log",
        distribution="poisson",
        fit_intercept=False,
        verbose=10,
        tol=1e-99,
    ).fit(X, y)

    # Compare coefficients
    print(poisson_sklearn.coef_, poisson_gam.coef_)
    print("sklearn", mean_poisson_deviance(y_true=y, y_pred=poisson_sklearn.predict(X)))
    print("gam", mean_poisson_deviance(y_true=y, y_pred=poisson_gam.predict(X)))

    # assert np.allclose(poisson_sklearn.coef_, poisson_gam.coef_)

    # Compare deviance
    dev_sklearn = mean_poisson_deviance(y_true=y, y_pred=poisson_sklearn.predict(X))
    dev_gam = mean_poisson_deviance(y_true=y, y_pred=poisson_gam.predict(X))
    assert dev_gam / dev_sklearn < 1.1
