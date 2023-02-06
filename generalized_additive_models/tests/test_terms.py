#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 09:18:35 2023

@author: tommy
"""
from generalized_additive_models.terms import Spline, TermList, Intercept, Linear
import numpy as np

import pytest
import itertools


class TestTermList:
    @pytest.mark.parametrize("element", [0, "a", [1, 2], Intercept])
    def test_that_only_terms_can_be_added(self, element):
        # An arbitrary object cannot be used in TermList
        with pytest.raises(TypeError):
            TermList(element)

        # A list with an arbitrary object cannot be used either
        with pytest.raises(TypeError):
            TermList([element, Intercept()])


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
            Linear(-1).transform(X)

        # The by-variable is too large
        with pytest.raises(ValueError):
            Linear(0, by=2).transform(X)

        # Negative penalty
        with pytest.raises(ValueError):
            Linear(0, by=1, penalty=-1).transform(X)

        # Negative penalty
        with pytest.raises(TypeError):
            Linear(0, by=1, penalty="asdf").transform(X)

        # Feature and by-variable is the same
        with pytest.raises(ValueError, match="cannot be equal to"):
            Linear(0, by=0).transform(X)


class TestSplines:
    @pytest.mark.parametrize(
        "by, num_splines, edges, periodic, degree, knots, extrapolation",
        list(
            itertools.product(
                [None, 1],
                [6, 12],
                [None, (0, 1)],
                [True, False],
                [0, 1, 2, 3, 4],
                ["uniform", "quantile"],
                ["constant", "linear", "continue", "periodic"],
            )
        ),
    )
    def test_that_number_of_splines_is_correct_for_all_inputs(
        self, by, num_splines, edges, periodic, degree, knots, extrapolation
    ):
        # Create a dummy matrix of data
        X = np.linspace(0, 1, num=128).reshape(-1, 2)
        print(X)

        # Create a spline
        spline = Spline(
            feature=0,
            by=by,
            num_splines=num_splines,
            edges=edges,
            periodic=periodic,
            degree=degree,
            knots=knots,
            extrapolation=extrapolation,
        )

        transformed_X = spline.transform(X)
        n_samples, n_features = transformed_X.shape
        assert n_samples == 64
        assert n_features == num_splines

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


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--capture=sys", "--doctest-modules", "--maxfail=1"])
