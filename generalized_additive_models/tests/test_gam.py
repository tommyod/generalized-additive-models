#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 07:11:09 2023

@author: tommy
"""
import itertools

import numpy as np
import pandas as pd
import pytest
import scipy as sp
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.utils import resample

from generalized_additive_models.distributions import Binomial, Normal, Poisson
from generalized_additive_models.gam import GAM
from generalized_additive_models.links import Identity, Log, Logit
from generalized_additive_models.terms import Categorical, Intercept, Linear, Spline, Tensor, TermList
from generalized_additive_models.links import Logit
from generalized_additive_models.distributions import Binomial

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


class TestAPIContract:
    def test_that_underscore_results_are_present(self):
        data = load_diabetes(as_frame=True)
        df = data.data
        y = data.target
        gam = GAM(Spline("age") + Spline("bmi") + Categorical("sex") + Linear("s5"))
        assert not hasattr(gam, "coef_")
        assert not hasattr(gam, "results_")
        assert not any(hasattr(t, "coef_") for t in gam.terms)
        assert not any(hasattr(t, "coef_indicies_") for t in gam.terms)
        gam.fit(df, y)

        # Test coefficients
        assert hasattr(gam, "coef_")
        assert isinstance(gam.coef_, np.ndarray)
        assert gam.coef_.ndim == 1
        assert np.abs(gam.coef_).mean() > 0

        # Test results_ Bunch
        assert hasattr(gam, "results_")
        assert hasattr(gam.results_, "covariance")
        assert hasattr(gam.results_, "edof_per_coef")
        assert hasattr(gam.results_, "edof")
        assert np.isclose(gam.results_.edof_per_coef.sum(), gam.results_.edof)

        # Test that attributes are copied over to terms
        assert all(hasattr(t, "coef_") for t in gam.terms)
        assert all(hasattr(t, "coef_idx_") for t in gam.terms)
        assert all(hasattr(t, "coef_covar_") for t in gam.terms)
        assert all(np.allclose(gam.coef_[term.coef_idx_], term.coef_) for term in gam.terms)
        for term in gam.terms:
            if isinstance(term, Categorical):
                assert hasattr(term, "categories_")
                assert len(term.categories_) == len(term.coef_)


class TestExponentialFunctionGamsWithCanonicalLinks:
    INTERCEPT = [-2, -1, 0, 1, 1.5]

    @pytest.mark.parametrize("intercept", INTERCEPT)
    def test_canonical_normal(self, intercept):
        rng = np.random.default_rng(1)

        # Create a normal problem
        x = np.linspace(0, 2 * np.pi, num=100_000)
        X = x.reshape(-1, 1)
        linear_prediction = intercept + np.sin(x)

        mu = Identity().inverse_link(linear_prediction)

        y = rng.normal(loc=mu, scale=0.05)

        # Create a GAM
        normal_gam = GAM(
            Spline(0, extrapolation="periodic"),
            link="identity",
            distribution="normal",
        ).fit(X, y)

        assert np.allclose(mu, normal_gam.predict(X), atol=0.01)

    @pytest.mark.parametrize("intercept", INTERCEPT)
    def test_canonical_poisson(self, intercept):
        rng = np.random.default_rng(2)

        # Create a poisson problem
        x = np.linspace(0, 2 * np.pi, num=100_000)
        X = x.reshape(-1, 1)
        linear_prediction = intercept + np.sin(x)

        mu = Log().inverse_link(linear_prediction)

        y = rng.poisson(lam=mu)

        # Create a GAM
        poisson_gam = GAM(
            Spline(0, extrapolation="periodic"),
            link="log",
            distribution="poisson",
        ).fit(X, y)

        assert np.allclose(mu, poisson_gam.predict(X), atol=0.1)

    @pytest.mark.parametrize("intercept", INTERCEPT)
    def test_caononical_logistic(self, intercept):
        rng = np.random.default_rng(3)

        # Create a logistic problem
        x = np.linspace(0, 2 * np.pi, num=100_000)
        X = x.reshape(-1, 1)
        linear_prediction = intercept + np.sin(x)

        mu = Logit().inverse_link(linear_prediction)

        y = rng.binomial(n=1, p=mu)

        # Create a GAM
        logistic_gam = GAM(
            Spline(0, extrapolation="periodic"),
            link="logit",
            distribution=Binomial(trials=1),
        ).fit(X, y)

        assert np.allclose(mu, logistic_gam.predict(X), atol=0.05)


class TestPandasCompatibility:
    @pytest.mark.parametrize("term_cls", [Spline, Linear])
    def test_that_string_columns_pose_no_problems(self, term_cls):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": list("qweras")})
        y = np.array([1, 2, 3, 4, 5, 6])

        gam = GAM(terms=term_cls("a"), fit_intercept=True)
        gam.fit(df, y)

    @pytest.mark.parametrize("term_cls", [Spline, Linear])
    def test_that_column_order_can_be_permuted_between_fit_and_transform(self, term_cls):
        rng = np.random.default_rng(3)
        col_A = np.exp(rng.normal(size=100))
        col_B = rng.normal(size=100)

        df = pd.DataFrame({"a": col_A, "b": col_B})

        # Transform column a
        term = term_cls("a")
        col_A_basis = term.fit_transform(df)

        # Now permute the columns
        df = df[["b", "a"]]

        # Transform column a again
        assert np.allclose(term.transform(df), col_A_basis)

    def test_that_integer_terms_can_be_used_with_pandas(self):
        # Get data as a DataFrame and Series
        data = fetch_california_housing(as_frame=True)
        df, y = data.data, data.target

        # Decrease data sets to speed up tests
        df, y = resample(df, y, replace=False, n_samples=100, random_state=1)
        df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=42)

        # Fit a model using column names
        gam = GAM(terms=Spline("AveRooms"), fit_intercept=True)
        gam.fit(df_train, y_train)

        # Fit a model using integer index
        gam2 = GAM(terms=Spline(2), fit_intercept=True)
        gam2.fit(df_train, y_train)

        assert np.allclose(gam.predict(df_test), gam2.predict(df_test))

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

    def test_cloning_with_sklearn_clone_non_default_link_and_distribution(self):
        gam = GAM(Spline(0), link=Logit(low=-10, high=10), distribution=Binomial(5))

        # Clone and change original
        cloned_gam = clone(gam)
        gam.link.low = 0
        gam.link.set_params(high=1)
        gam.distribution.trials = 100

        # Check clone
        assert cloned_gam.link.low == -10
        assert cloned_gam.link.high == 10
        assert cloned_gam.distribution.trials == 5

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


class TestGamAutoModels:
    """
    Auto models means setting

    >>> penalty = 2
    >>> GAM(Spline(None, penalty=penalty))
    GAM(terms=Spline(penalty=2))

    This will expand the terms to an all-spline model when data is seen.
    """

    @pytest.mark.parametrize(
        "penalty, fit_intercept, term_class",
        list(itertools.product([0.01, 0.1, 1, 10, 100], [True, False], [Spline, Linear])),
    )
    def test_auto_model(self, penalty, fit_intercept, term_class):
        X, y = fetch_california_housing(return_X_y=True, as_frame=False)

        # Decrease data sets to speed up test
        X, y = resample(X, y, replace=False, n_samples=1000, random_state=42)
        num_samples, num_features = X.shape

        # Auto model
        auto_gam = GAM(term_class(None, penalty=penalty), fit_intercept=fit_intercept).fit(X, y)

        # Manual model
        terms = TermList(term_class(i, penalty=penalty) for i in range(num_features))
        manual_gam = GAM(terms, fit_intercept=fit_intercept).fit(X, y)

        assert auto_gam.terms == manual_gam.terms
        assert np.allclose(auto_gam.predict(X), manual_gam.predict(X))

    def test_that_auto_model_with_grid_search_CV_selects_good_model(self):
        # Get data
        data = fetch_california_housing(as_frame=True)
        df = data.data.iloc[:, :-2]  # Remove spatial columns
        y = data.target

        # Decrease data sets to speed up test
        df, y = resample(df, y, replace=False, n_samples=1000, random_state=42)

        # Createa model and create object
        gam = GAM(Spline(None))
        param_grid = {
            "terms__penalty": np.logspace(-3, 3, num=7),
            "terms__extrapolation": ["linear", "constant", "continue"],
        }
        search = GridSearchCV(gam, param_grid, scoring="r2", n_jobs=-1)

        search.fit(df, y)

        assert search.best_score_ > 0.6
        assert search.best_params_["terms__penalty"] > 1e-3
        assert search.best_params_["terms__penalty"] < 1e3


class TestGAMSanityChecks:
    @pytest.mark.parametrize("mean_value", [-100, -10, 0, 10, 100, 1000, 10000, 100000])
    def test_that_mean_value_is_picked_up_by_intercept(self, mean_value):
        # Create a 1D data set y = x * log(x) on x \in (0, 2)
        rng = np.random.default_rng(42)
        X = rng.random(size=(10_000, 1)) * np.pi * 2
        y = np.sin(X.ravel())

        y = y.ravel() - np.mean(y) + mean_value

        # Create a GAM and fit it
        gam = GAM(terms=Spline(0, num_splines=20, degree=3, penalty=1), fit_intercept=True)

        gam.fit(X, y)

        for term in gam.terms:
            if isinstance(term, Intercept):
                assert np.isclose(term.coef_, mean_value, rtol=0.01)

    def test_that_tensors_outperform_splines_on_multiplicative_problem(self):
        # Create a data set which is hard for an additive model to predict
        # y = exp(-x**2 - y**2)
        rng = np.random.default_rng(42)
        X = rng.random(size=(1000, 2)) * 2 - 1
        y = np.exp(-3 * np.linalg.norm(X, axis=1) ** 2)

        # Set up CV object
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Linear model GAM, with no-so-good performance
        gam_spline_model = GAM(Spline(0, penalty=10) + Spline(1, penalty=10) + Intercept())
        score_spline_model = cross_val_score(gam_spline_model, X, y, verbose=0, cv=cv).mean()

        # Tensor model GAM, with better performance
        gam_tensor_model = GAM(Tensor([Spline(0, penalty=10), Spline(1, penalty=10)]) + Intercept())
        score_tensor_model = cross_val_score(gam_tensor_model, X, y, verbose=0, cv=cv).mean()

        assert score_tensor_model > score_spline_model
        assert score_tensor_model > 0.95
        assert score_spline_model < 0.5

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

    def test_logistic_gam_on_breast_cancer_dataset(self):
        # Load data
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Train GAM using automodel feature (sending in one spline and expanding)
        terms = TermList(Spline(None, extrapolation="continue", num_splines=8, penalty=100))
        gam = GAM(terms, link="logit", distribution=Binomial(trials=1))
        gam.fit(X_train, y_train)
        gam_preds = gam.predict(X_test) > 0.5
        gam_accuracy = accuracy_score(y_true=y_test, y_pred=gam_preds)

        assert gam_accuracy > 0.95


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            "--capture=sys",
            "--doctest-modules",
            "--maxfail=1",
            # "-k TestExponentialFunctionGamsWithCanonicalLinks",
        ]
    )
    if False:
        rng = np.random.default_rng(2)

        # Create a poisson problem
        x = np.linspace(0, 2 * np.pi, num=100_000)
        linear_prediction = 0.5 + np.sin(x)
        mu = Log().inverse_link(linear_prediction)
        y = rng.poisson(lam=mu)
        X = x.reshape(-1, 1)

        # Create a GAM
        poisson_gam = GAM(
            Spline(0, extrapolation="periodic"),
            link="log",
            distribution="poisson",
        ).fit(X, y)

        import matplotlib.pyplot as plt

        # plt.plot(mu, poisson_gam.predict(X))

        plt.plot(x, mu)
        plt.plot(x, poisson_gam.predict(X))
