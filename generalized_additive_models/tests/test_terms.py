#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 09:18:35 2023

@author: tommy
"""
from sklearn.base import clone
from generalized_additive_models.terms import Spline, TermList, Intercept, Linear, Tensor, Term
import numpy as np
import itertools

import pytest
import itertools


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


class TestTermList:
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
