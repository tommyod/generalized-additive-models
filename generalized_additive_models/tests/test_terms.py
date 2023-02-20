#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 09:18:35 2023

@author: tommy
"""
import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing
from sklearn.exceptions import NotFittedError

from generalized_additive_models.terms import Categorical, Intercept, Linear, Spline, Tensor, Term, TermList
from generalized_additive_models.gam import GAM


class TestTermMultiplications:
    """Tests for modeling algebra."""

    @pytest.mark.skip(reason="Not implemented.")
    def test_intercept_multiplications(self):
        redundant = Term.is_redudance_with_respect_to

        i = Intercept()
        assert i * i == i

        l = Linear(0)
        assert l * i == i * l == l
        assert redundant(l * i, i * l)

        s = Spline(0, by=1)
        assert s * i == i * s == s
        assert redundant(s * i, i * s)

        te = Tensor([Spline(0), Spline(1)])
        assert te * i == i * te == te
        assert redundant(te * i, i * te)

    @pytest.mark.skip(reason="Not implemented.")
    def test_linear_multiplication(self):
        redundant = Term.is_redudance_with_respect_to

        l = Linear(0)
        l2 = Linear(1)

        assert l * l2 == Linear(1, by=0) == Linear(0, by=1) == l2 * l
        assert redundant(l * l2, Linear(1, by=0))
        assert redundant(l * l2, Linear(0, by=1))
        assert redundant(l * l2, l2 * l)

        s = Spline(1)
        assert l * s == s * l == Spline(1, by=0)

        te = Tensor([Spline(1), Spline(2)])
        assert l * te == te * l == Tensor([Spline(1), Spline(2)], by=0)

    @pytest.mark.skip(reason="Not implemented.")
    def test_spline_multiplication(self):
        s1 = Spline(1)
        s2 = Spline(2)
        assert s1 * s2 == s2 * s1 == Tensor([s1, s2])

        te = Tensor([Spline(3), Spline(4)])
        assert s1 * te == te * s1 == Tensor([Spline(0), Spline(3), Spline(4)])

    @pytest.mark.skip(reason="Not implemented.")
    def test_tensor_multiplication(self):
        te12 = Tensor([Spline(1), Spline(2)])
        te34 = Tensor([Spline(2), Spline(4)])
        assert te12 * te34 == te34 * te12 == Tensor([Spline(1), Spline(2), Spline(3), Spline(4)])

    @pytest.mark.skip(reason="Not implemented.")
    def test_non_associative_multiplication(self):
        l0 = Linear(0)
        s1 = Spline(1)
        s2 = Spline(2)

        assert l0 * (s1 * s2) == Tensor([Spline(1), Spline(2)], by=0)
        assert (l0 * s1) * s2 == Tensor([Spline(1, by=0), Spline(2)])

    @pytest.mark.skip(reason="Not implemented.")
    def test_tensor_cyclic_property(self):
        te123 = Tensor([Spline(1), Spline(2), Spline(3)])

        for permutation in itertools.permutations([1, 2, 3]):
            assert te123 == Tensor([Spline(i) for i in permutation])

    @pytest.mark.skip(reason="Not implemented.")
    def test_distributivity_over_terms(self):
        i = Intercept()
        l = Linear(0)
        s = Spline(1)
        te = Tensor([Spline(2), Spline(3)])

        assert i * (l + s + te) == (i * l + i * s + i * te) == (l + s + te) * i
        assert l * (i + s + te) == l * i + l * s + l * te == (i + s + te) * l
        assert s * (i + l + te) == s * i + s * l + s * te == (i + l + te) * s
        assert te * (i + l + s) == te * i + te * l + te * s == (i + l + s) * te


class TestTerms:
    # check_is_fitted(self, attributes=["coef_"])

    @pytest.mark.parametrize("term", [Intercept, Linear, Spline, Categorical, Tensor])
    def test_that_bounds_are_set(self, term):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(100, 2))

        if term in (Linear, Spline, Categorical):
            term = term(0)
        elif term in (Intercept,):
            term = term()
        elif term in (Tensor,):
            term = term([Spline(0), Spline(1)])
        else:
            assert False

        term = term.fit(X)
        assert hasattr(term, "_bounds")
        assert len(term._bounds) == term.num_coefficients

    @pytest.mark.parametrize("term", [Intercept, Linear, Spline, Categorical, Tensor])
    def test_that_transform_fails_if_not_fitted(self, term):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(100, 2))

        if term in (Linear, Spline, Categorical):
            term = term(0)
        elif term in (Intercept,):
            term = term()
        elif term in (Tensor,):
            term = term([Spline(0), Spline(1)])
        else:
            assert False

        with pytest.raises(NotFittedError):
            term.transform(X)

    def test_that_linear_can_infer_mean(self):
        X = np.linspace(-1, 1, num=101).reshape(-1, 1)
        y = np.e + np.pi * X.ravel()

        linear = Linear(0, penalty=0)
        intercept = Intercept()
        gam = GAM(linear + intercept)
        gam.fit(X, y)

        assert np.allclose(linear.coef_, np.pi)
        assert np.allclose(intercept.coef_, np.e)

    def test_that_categorical_can_infer_means(self):
        # Group 1 has mean 3, group 2 has mean 5
        group_means = np.array([3, 5])
        df = pd.DataFrame({"group": [1, 1, 1, 1, 1, 2, 2], "value": [2, 3, 4, 2, 4, 4, 6]})
        categorical = Categorical("group", penalty=1e-4)  # Decrease regularization
        intercept = Intercept()

        gam = GAM(categorical + intercept)
        gam.fit(df, df.value)

        predicted_group_means = categorical.coef_ + intercept.coef_
        assert np.allclose(predicted_group_means, group_means)
        assert np.allclose(intercept.coef_, 4)  # mean([3, 5]) = 4

    def test_categorical_regularization_with_high_penalty(self):
        # Group 1 has mean 3, group 2 has mean 5
        df = pd.DataFrame({"group": [1, 1, 1, 1, 1, 2, 2], "value": [2, 3, 4, 2, 4, 4, 6]})
        grand_mean = df.value.mean()

        categorical = Categorical("group", penalty=1e6)  # Increase regularization
        intercept = Intercept()

        gam = GAM(categorical + intercept)
        gam.fit(df, df.value)

        predicted_group_means = categorical.coef_ + intercept.coef_
        assert np.allclose(predicted_group_means, grand_mean)

    def test_that_feature_and_by_are_symmetric_for_linear_term(self):
        rng = np.random.default_rng(32)
        X = rng.normal(size=(100, 2))

        term1 = Linear(0, by=1)
        term2 = Linear(1, by=0)

        assert np.allclose(term1.fit_transform(X), term2.fit_transform(X))

    @pytest.mark.parametrize("term", [Linear, Spline])
    def test_that_transformed_data_sums_to_zero_in_each_column(self, term):
        rng = np.random.default_rng(33)
        X = rng.normal(size=(100, 2))

        # Convert to integer to get meaningful categories
        if isinstance(term, Categorical):
            X[:, 0] = np.asarray(X[:, 0], dtype=int)

        term = term(0, by=1)
        X_transformed = term.fit_transform(X)
        assert np.allclose(X_transformed.sum(axis=0), 0)

    @pytest.mark.parametrize("constraint", ["increasing", "decreasing", "convex", "concave"])
    def test_that_transformed_data_sums_to_zero_in_each_column_with_constraints(self, constraint):
        rng = np.random.default_rng(33)
        X = rng.normal(size=(100, 2))

        term = Spline(0, by=1, constraint=constraint)
        X_transformed = term.fit_transform(X)
        assert np.allclose(X_transformed.sum(axis=0), 0)

    def test_that_transformed_data_sums_to_zero_in_each_column_in_tensor(self):
        rng = np.random.default_rng(33)
        X = rng.normal(size=(100, 5))

        # With no 'by' variables in the terms
        term = Tensor([Spline(0), Spline(1)], by=2)
        X_transformed = term.fit_transform(X)
        assert np.allclose(X_transformed.sum(axis=0), 0)

        # With no 'by' variable in one of the terms
        term = Tensor([Spline(0, by=3), Spline(1)], by=2)
        X_transformed = term.fit_transform(X)
        assert np.allclose(X_transformed.sum(axis=0), 0)

        # With 'by' variables in both of the terms
        term = Tensor([Spline(0, by=3), Spline(1, by=4)], by=2)
        X_transformed = term.fit_transform(X)
        assert np.allclose(X_transformed.sum(axis=0), 0)

    @pytest.mark.parametrize("term", [Linear, Spline])
    def test_that_transforming_a_single_column_does_not_forget_feature_index(self, term):
        x1 = np.arange(9)
        x2 = np.arange(9) + 5
        X = np.vstack((x1, x2)).T

        term = term(1)
        transformed = term.fit_transform(X)

        # Transforming a 1-column matrix works, even if it's not a dataframe
        # with the same columns as was fitted on
        transformed_other = term.transform(x1.reshape(-1, 1))

        # Different results due to different data
        assert not np.allclose(transformed, transformed_other)

        # The term is still able to transform the same column and get the same result
        transformed2 = term.transform(X)
        assert np.allclose(transformed, transformed2)


class TestTermList:
    def test_that_adding_tensor_term_does_not_unpack_tensor(self):
        # Since a Tensor is iterable, a bug caused each individual
        # spline in the Tensor to be added, instead of the tensor

        # Check for each permutation
        terms = Spline(0) + Spline(1) + Tensor([Spline(2), Spline(3)])
        assert any(isinstance(term, Tensor) for term in terms)

        terms = Spline(1) + Tensor([Spline(2), Spline(3)]) + Spline(0)
        assert any(isinstance(term, Tensor) for term in terms)

        terms = Tensor([Spline(2), Spline(3)]) + Spline(0) + Spline(1)
        assert any(isinstance(term, Tensor) for term in terms)

    @pytest.mark.parametrize("element", [0, "a", [1, 2], Intercept])
    def test_that_only_terms_can_be_added(self, element):
        # An arbitrary object cannot be used in TermList
        with pytest.raises(TypeError):
            TermList(element)

        # A list with an arbitrary object cannot be used either
        with pytest.raises(TypeError):
            TermList([element, Intercept()])

    def test_cloning_with_sklearn_clone(self):
        term_list = TermList(Spline(0) + Spline(1))
        cloned_term_list = clone(term_list)
        assert term_list == cloned_term_list
        assert term_list is not cloned_term_list

        # A more intricate example
        s = Spline(0, penalty=1)
        term_list = TermList([s, Intercept()])
        cloned_term_list = clone(term_list)

        # Changing the original term should not change the clone
        s.set_params(penalty=99)
        assert cloned_term_list[0].penalty == 1

        # But the original changed
        assert term_list[0].penalty == 99


class TestPenaltyMatrices:
    @pytest.mark.parametrize("num_splines", [5, 10, 15])
    def test_that_penalty_matrix_shape_is_correct(self, num_splines):
        terms = [Intercept(), Linear(0), Spline(0, num_splines=num_splines)]
        for term in terms:
            assert term.num_coefficients > 0

            # Matrix must be returned
            assert term.penalty_matrix().ndim == 2

            # Matrix multiplication of penalty against coeffs must be correct
            assert term.penalty_matrix().shape[1] == term.num_coefficients

        spline = Spline(0, num_splines=num_splines)
        assert spline.num_coefficients == spline.num_splines


class TestTermParameters:
    def test_that_invalid_parameters_raise_in_linear_term(self):
        X = np.linspace(0, 1).reshape(-1, 2)

        # Invalid feature
        with pytest.raises(ValueError):
            Linear(-1).fit_transform(X)

        # The by-variable is too large
        with pytest.raises(ValueError):
            Linear(0, by=2).fit_transform(X)

        # Negative penalty
        with pytest.raises(ValueError):
            Linear(0, by=1, penalty=-1).fit_transform(X)

        # Negative penalty
        with pytest.raises(TypeError):
            Linear(0, by=1, penalty="asdf").fit_transform(X)

        # Feature and by-variable is the same
        with pytest.raises(ValueError, match="cannot be equal to"):
            Linear(0, by=0).fit_transform(X)


class TestPandasCompatibility:
    @pytest.mark.parametrize("term", [Linear, Spline])
    def test_that_terms_can_use_numerical_and_string_features(self, term):
        # Load data as dataframe
        df = fetch_california_housing(as_frame=True).data
        assert isinstance(df, pd.DataFrame)

        # Load as numpy array
        X, _ = fetch_california_housing(return_X_y=True)
        assert isinstance(X, np.ndarray)

        assert df.shape == X.shape

        # Check that the results are the same
        for i, feature in enumerate(df.columns):
            assert np.allclose(term(feature=feature).fit_transform(df), term(feature=i).fit_transform(X))

    @pytest.mark.parametrize("term", [Linear, Spline])
    def test_that_transforming_a_single_column_does_not_forget_feature_name(self, term):
        x1 = np.arange(9)
        x2 = np.arange(9) + 5
        df = pd.DataFrame({"x1": x1, "x2": x2})

        term = term("x1")
        transformed = term.fit_transform(df)

        # Transforming a 1-column matrix works, even if it's not a dataframe
        # with the same columns as was fitted on
        transformed_other = term.transform(x2.reshape(-1, 1))

        # Different results due to different data
        assert not np.allclose(transformed, transformed_other)

        # The term is still able to transform the same column and get the same result
        transformed2 = term.transform(df)
        assert np.allclose(transformed, transformed2)


class TestSplines:
    @pytest.mark.parametrize(
        "by, num_splines, edges, degree, knots, extrapolation",
        list(
            itertools.product(
                [None, 1],
                [6, 12],
                [None, (0, 1)],
                [0, 1, 2, 3, 4],
                ["uniform", "quantile"],
                ["constant", "linear", "continue", "periodic"],
            )
        ),
    )
    def test_that_number_of_splines_is_correct_for_all_inputs(
        self, by, num_splines, edges, degree, knots, extrapolation
    ):
        # Create a dummy matrix of data
        X = np.linspace(0, 1, num=128).reshape(-1, 2)

        # Create a spline
        spline = Spline(
            feature=0,
            by=by,
            num_splines=num_splines,
            edges=edges,
            degree=degree,
            knots=knots,
            extrapolation=extrapolation,
        )

        transformed_X = spline.fit_transform(X)
        n_samples, n_features = transformed_X.shape
        assert n_samples == 64
        assert n_features == spline.num_coefficients
        assert n_features == spline.num_splines

    def test_that_setting_and_getting_works_on_intercept(self):
        intercept = Intercept()
        assert intercept.get_params() == Intercept().set_params(**intercept.get_params()).get_params()
        assert intercept.get_params() == Intercept(**intercept.get_params()).get_params()

    @pytest.mark.parametrize(
        "feature, penalty, by",
        list(itertools.product([None, 0, 1, 2], [0.1, 1, 10], [None, 3, 4, 5])),
    )
    def test_that_setting_and_getting_works_on_linear(self, feature, penalty, by):
        linear = Linear(feature=feature, penalty=penalty, by=by)
        assert linear.get_params() == Linear().set_params(**linear.get_params()).get_params()
        assert linear.get_params() == Linear(**linear.get_params()).get_params()

    def test_cloning_with_sklearn_clone(self):
        spline = Spline(0, penalty=999)
        cloned_spline = clone(spline)
        assert cloned_spline == spline

        spline.set_params(penalty=1)
        assert spline.penalty == 1
        assert cloned_spline.penalty == 999

    def test_spline_transformations_and_penalties(self):
        pass


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--capture=sys", "--doctest-modules", "--maxfail=1"])
