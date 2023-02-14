#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 07:11:09 2023

@author: tommy
"""
import pytest
import numpy as np
from sklearn.base import clone
from generalized_additive_models.gam import GAM
from generalized_additive_models.terms import Spline, Linear, Intercept, Tensor, TermList
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import scipy as sp
from sklearn.utils import resample

SMOOTH_FUNCTIONS = [
    np.log1p,
    np.exp,
    np.sin,
    np.cos,
    np.cosh,
    np.sinc,
    np.sqrt,
    np.square,
    sp.special.expm1,
    sp.special.expit,
]


class TestPandasCompatibility:
    def test_models_of_increasing_complexity_from_a_pandas_dataframe(self):
        # Get data as a DataFrame and Series
        data = fetch_california_housing(as_frame=True)
        df, y = data.data, data.target

        # Decrease data sets to speed up tests
        df, y = resample(df, y, replace=False, n_samples=1000, random_state=42)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(y, pd.Series)

        df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=42)
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)

        # Fit a constant model
        gam = GAM(terms=Intercept(), fit_intercept=False)
        gam.fit(df_train, y_train)
        score = r2_score(y_true=y_test, y_pred=gam.predict(df_test))
        assert np.isclose(score, 0, atol=0.05)

        # Fit a linear model
        gam = GAM(terms=TermList(Linear(c) for c in df.columns), fit_intercept=True)
        gam.fit(df_train, y_train)
        score = r2_score(y_true=y_test, y_pred=gam.predict(df_test))
        assert np.isclose(score, 0.6, atol=0.1)

        # Fit a spline model
        gam = GAM(terms=TermList(Spline(c) for c in df.columns), fit_intercept=True)
        gam.fit(df_train, y_train)
        score = r2_score(y_true=y_test, y_pred=gam.predict(df_test))
        assert np.isclose(score, 0.65, atol=0.1)

    def test_that_dataframe_and_numpy_produce_idential_results(self):
        # Get data as a DataFrame and Series
        data = fetch_california_housing(as_frame=True)
        df, y = data.data, data.target

        # Decrease data sets to speed up tests
        df, y = resample(df, y, replace=False, n_samples=100, random_state=42)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(y, pd.Series)

        columns_to_use = np.array([0, 3, 5])

        # Fit a model using DataFrame
        terms = TermList(Spline(c) for c in df.columns[columns_to_use])
        gam1 = GAM(terms=terms, fit_intercept=True)
        gam1.fit(df, y)

        # Fit a model using numpy array
        terms = TermList(Spline(c) for c in columns_to_use)
        gam2 = GAM(terms=terms, fit_intercept=True)
        gam2.fit(df.values, y.values)

        assert np.allclose(gam1.predict(df), gam2.predict(df.values))


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

        # Decrease data sets to speed up test
        X, y = resample(X, y, replace=False, n_samples=1000, random_state=42)

        gam = GAM(terms=Spline(0))

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(gam, X, y, verbose=0, cv=cv)

        assert scores.mean() > 0.4

    def test_that_sklearn_grid_search_works(self):
        X, y = fetch_california_housing(return_X_y=True, as_frame=False)

        # Decrease data sets to speed up test
        X, y = resample(X, y, replace=False, n_samples=1000, random_state=42)

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

        # Decrease data sets to speed up test
        X, y = resample(X, y, replace=False, n_samples=1000, random_state=42)

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
    @pytest.mark.skip("Mean will not equal mean of data")
    @pytest.mark.parametrize("mean_value", [-100, -10, 0, 10, 100, 1000, 10000])
    def test_that_mean_value_is_picked_up_by_intercept(self, mean_value):
        # Create a 1D data set y = x * log(x) on x \in (0, 2)
        rng = np.random.default_rng(42)
        X = rng.random(size=(1000, 1)) * 2
        X = np.sort(X, axis=0)

        y = np.log(X) * X
        y = y.ravel() - np.mean(y) + mean_value

        # Create a GAM and fit it
        gam = GAM(terms=Spline(0, num_splines=20, degree=3, penalty=1), fit_intercept=True)

        gam.fit(X, y)

        for term in gam.terms:
            if isinstance(term, Intercept):
                assert np.isclose(term.coef_, mean_value)

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

    @pytest.mark.parametrize("function", SMOOTH_FUNCTIONS)
    def test_that_1D_spline_score_on_smooth_function_is_close(self, function):
        rng = np.random.default_rng(42)
        X = rng.random(size=(1000, 1)) * 5
        y = function(X.ravel())

        # Create a GAM
        linear_gam = GAM(Spline(0), fit_intercept=True)
        linear_gam.fit(X, y)
        assert linear_gam.score(X, y) > 0.99

    @pytest.mark.parametrize("function", SMOOTH_FUNCTIONS)
    def test_that_1D_spline_score_on_smooth_function_with_by_is_close(self, function):
        # Create data of form: y = f(x_1) * x_2
        rng = np.random.default_rng(42)
        X = rng.random(size=(1000, 2)) * np.pi
        y = function(X[:, 0]) * X[:, 1]

        # Create a GAM and fit it
        gam = GAM(terms=Spline(0, by=1), fit_intercept=True)
        gam.fit(X, y)
        gam_score = gam.score(X, y)
        assert gam_score > 0.99

        # Bad gam
        bad_gam = GAM(terms=Spline(0) + Linear(1), fit_intercept=True)
        bad_gam.fit(X, y)
        bad_gam_score = bad_gam.score(X, y)
        assert bad_gam_score < gam_score


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])
