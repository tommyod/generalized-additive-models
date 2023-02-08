#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 07:11:09 2023

@author: tommy
"""
import numpy as np
from sklearn.base import clone
from generalized_additive_models.gam import GAM
from generalized_additive_models.terms import Spline, Linear, Intercept, Tensor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


class TestSklearnCompatibility:
    def test_cloning_with_sklearn_clone(self):
        terms = Spline(0, extrapolation="periodic")
        gam = GAM(terms=terms, max_iter=100)

        # Clone and change original
        cloned_gam = clone(gam)
        gam.max_iter = 1
        gam.set_params(terms=Spline(0, extrapolation="linear"))

        # Check clone
        assert cloned_gam.max_iter == 100
        assert cloned_gam.terms.extrapolation == "periodic"

    def test_that_get_and_set_params_works(self):
        terms = Spline(0, extrapolation="periodic")
        gam = GAM(terms=terms, max_iter=100)

        assert {
            "terms": Spline(feature=0, extrapolation="periodic"),
            "max_iter": 100,
        }.items() <= gam.get_params(False).items()

        assert {
            "max_iter": 100,
            "terms__feature": 0,
            "terms__extrapolation": "periodic",
            "terms": Spline(feature=0, extrapolation="periodic"),
        }.items() <= gam.get_params(True).items()

    def test_that_sklearn_cross_val_score_works(self):
        X, y = fetch_california_housing(return_X_y=True, as_frame=False)

        gam = GAM(terms=Spline(0))

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(gam, X, y, verbose=0, cv=cv)

        assert scores.mean() > 0.4

    def test_that_sklearn_grid_search_works(self):
        X, y = fetch_california_housing(return_X_y=True, as_frame=False)

        gam = GAM(terms=Intercept(), fit_intercept=True)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            gam,
            param_grid={"terms": [Intercept(), Linear(0), Spline(0)]},
            scoring=None,
            n_jobs=1,
            refit=True,
            cv=cv,
            verbose=99,
            pre_dispatch="2*n_jobs",
            return_train_score=False,
        )

        search.fit(X, y)

        assert search.best_params_ == {"terms": Spline(feature=0)}
        assert search.best_score_ > 0.4

    def test_that_sklearn_grid_search_works_over_penalties(self):
        X, y = fetch_california_housing(return_X_y=True, as_frame=False)

        gam = GAM(terms=Spline(0, penalty=0) + Intercept(), fit_intercept=False)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        search = GridSearchCV(
            gam,
            param_grid={"terms__0__penalty": [0.01, 0.1, 1, 10, 100, 1000]},
            scoring=None,
            n_jobs=1,
            cv=cv,
        )

        search.fit(X, y)
        assert search.best_score_ > 0.45


class TestGAMSanityChecks:
    def test_that_tensors_outperform_splines_on_multiplicative_problem(self):
        # Create a data set which is hard for an additive model to predict
        # y = exp(-x**2 - y**2)
        rng = np.random.default_rng(42)
        X = rng.random(size=(1000, 2)) * 2 - 1
        y = np.exp(-3 * np.linalg.norm(X, axis=1) ** 2)

        # Set up CV object
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Linear model GAM, with no-so-good performance
        gam_linear_model = GAM(Spline(0) + Spline(1) + Intercept(), fit_intercept=False)
        score_linear_model = cross_val_score(gam_linear_model, X, y, verbose=0, cv=cv).mean()

        # Spline model GAM, with better performance
        gam_spline_model = GAM(Tensor([Spline(0), Spline(1)]) + Intercept(), fit_intercept=False)
        score_spline_model = cross_val_score(gam_spline_model, X, y, verbose=0, cv=cv).mean()

        assert score_spline_model > score_linear_model
        assert score_spline_model > 0.98
        assert score_linear_model < 0.85


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])
