#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 09:18:35 2023

@author: tommy





"""
import numpy as np
import scipy as sp
import functools
from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from collections import UserList
import copy
from sklearn.preprocessing import SplineTransformer
from sklearn.base import TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import _get_feature_names
from numbers import Integral, Real
from generalized_additive_models.penalties import second_order_finite_difference
from generalized_additive_models.utils import tensor_product

from abc import ABC, abstractmethod
from collections.abc import Container
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


class Term(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @property
    @abstractmethod
    def num_coefficients(self):
        pass

    @abstractmethod
    def penalty_matrix(self):
        pass

    def infer_feature_variable(self, *, variable_name, X):
        num_samples, num_features = X.shape
        variable_content = getattr(self, variable_name)
        variable_to_set = f"{variable_name}_"

        # Set the self.feature_ variable
        if isinstance(variable_content, str):
            feature_names = _get_feature_names(X)  # None or np.array
            feature_names = [] if feature_names is None else list(feature_names)
            if variable_content not in feature_names:
                msg = f"Feature in {self} does not match feature names in the data: {feature_names}."
                raise ValueError(msg)
            else:
                setattr(self, variable_to_set, feature_names.index(variable_content))
        elif isinstance(variable_content, Integral):
            if variable_content not in range(0, num_features):
                raise ValueError(f"Parameter {self.feature=} must be in range [0, {num_features}].")
            else:
                # Copy it over
                setattr(self, variable_to_set, variable_content)

    def is_redudance_with_respect_to(self, other):
        """Check if a feature is redundance with respect to another feature.

        Examples
        --------
        >>> Spline(0).is_redudance_with_respect_to(Spline(0))
        True
        >>> Spline(0).is_redudance_with_respect_to(Spline(0, by=1))
        False
        >>> Intercept().is_redudance_with_respect_to(Intercept())
        True
        >>> Linear(0).is_redudance_with_respect_to(Linear(1))
        False
        >>> Linear(0).is_redudance_with_respect_to(Linear(0))
        True
        >>> te1 = Tensor([Spline(0), Spline(1)])
        >>> te2 = Tensor([Spline(1), Spline(0)])
        >>> te1.is_redudance_with_respect_to(te2)
        True
        >>> te2 = Tensor([Spline(1), Spline(0, by=1)])
        >>> te1.is_redudance_with_respect_to(te2)
        False

        """
        # Check if equal type
        equal_type = type(self) is type(other)
        if not equal_type:
            return False

        # Check for Spline/Linear/Intercept, etc
        if isinstance(self, (Intercept, Linear, Spline)):
            return frozenset([self.feature, self.by]) == frozenset([other.feature, other.by])
        # Check for Tensor
        elif isinstance(self, Tensor):
            self_vars = frozenset([frozenset([term.feature, term.by]) for term in self] + [self.by])
            other_vars = frozenset([frozenset([term.feature, term.by]) for term in other] + [other.by])
            return self_vars == other_vars
        else:
            raise TypeError(f"Cannot compare {self} and {other}")

    def __eq__(self, other):
        # Two terms are equal if their parameters are equal
        equal_types = type(self) == type(other)
        if not equal_types:
            return False

        equal_params = self.get_params() == other.get_params()
        return equal_params

    def __add__(self, other):
        return TermList(self) + TermList(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iter__(self):
        self._has_yielded = False
        return self

    def __next__(self):
        if self._has_yielded:
            del self._has_yielded
            raise StopIteration
        else:
            self._has_yielded = True
            return self


class Intercept(TransformerMixin, Term, BaseEstimator):
    name = "intercept"
    feature = None
    by = None

    # =============================================================================
    #     def __mul__(self, other):
    #         if not isinstance(other, Term):
    #             raise NotImplementedError
    #         return copy.deepcopy(other)
    #
    #     def __rmul__(self, other):
    #         return self.__mul__(other)
    # =============================================================================

    @property
    def num_coefficients(self):
        return 1

    def penalty_matrix(self):
        return np.array([[0.0]])

    def fit(self, X):
        return self

    def transform(self, X):
        """transform the term.


        Parameters
        ----------
        X : np.ndarray
            An ndarray with 2 dimensions of shape (n_samples, n_features).

        Returns
        -------
        X : np.ndarray
            An ndarray for the term.

        Examples
        --------
        >>> intercept = Intercept()
        >>> X = np.eye(3)
        >>> intercept.transform(X)
        array([[1.],
               [1.],
               [1.]])

        Terms can yield themselves once, like so:

        >>> list(intercept)
        [Intercept()]

        """
        X = check_array(X, estimator=self, input_name="X")
        n_samples, n_features = X.shape
        return np.ones(n_samples).reshape(-1, 1)


class Linear(TransformerMixin, Term, BaseEstimator):
    name = "linear"

    _parameter_constraints = {
        "feature": [Interval(Integral, 0, None, closed="left"), str, None],
        "penalty": [Interval(Real, 0, None, closed="left")],
        "by": [Interval(Integral, 0, None, closed="left"), str, None],
    }

    def __init__(self, feature=None, penalty=1, by=None):
        """Create a linear term with a given penalty.

        Examples
        --------
        >>> linear_term = Linear(0, penalty=2)
        >>> linear_term
        Linear(feature=0, penalty=2)
        >>> Linear(0, penalty=2) == Linear(0, penalty=2)
        True
        >>> Linear(0, penalty=2) == Linear(0, penalty=3)
        False
        >>> Linear(0, penalty=2).is_redudance_with_respect_to(Linear(0, penalty=3))
        True
        """
        self.feature = feature
        self.penalty = penalty
        self.by = by

    def _validate_params(self, X):
        # Validate using BaseEsimator._validate_params, which in turn calls
        # sklearn.utils._param_validation.validate_parameter_constraints
        # using the `_parameter_constraints` attributed defined on the class.
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/base.py#L573
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/utils/_param_validation.py#L28
        super()._validate_params()

        if self.by == self.feature:
            raise ValueError(f"Parameter {self.by=} cannot be equal to {self.feature=}")

        self.infer_feature_variable(variable_name="feature", X=X)
        self.infer_feature_variable(variable_name="by", X=X)

    @property
    def num_coefficients(self):
        return 1

    def penalty_matrix(self):
        super()._validate_params()  # Validate the 'penalty' parameter
        return np.sqrt(self.penalty) * np.array([[1.0]])

    def fit(self, X):
        self._validate_params(X)
        X = check_array(X, estimator=self, input_name="X")
        return self

    def transform(self, X):
        """transform the term.

        Parameters
        ----------
        X : np.ndarray
            An ndarray with 2 dimensions of shape (n_samples, n_features).

        Returns
        -------
        X : np.ndarray
            An ndarray for the term.

        Examples
        --------
        >>> linear = Linear(1)
        >>> X = np.eye(3)
        >>> linear.transform(X)
        array([[0.],
               [1.],
               [0.]])

        """
        self._validate_params(X)
        X = check_array(X, estimator=self, input_name="X")
        num_samples, num_features = X.shape

        basis_matrix = X[:, self.feature_].reshape(-1, 1)

        if self.by is not None:
            basis_matrix *= X[:, self.by_][:, np.newaxis]

        return basis_matrix


class Spline(TransformerMixin, Term, BaseEstimator):
    name = "spline"

    L2_penalty = 0.0

    _parameter_constraints = {
        "feature": [Interval(Integral, 0, None, closed="left"), str, None],
        "penalty": [Interval(Real, 0, None, closed="left")],
        "by": [Interval(Integral, 0, None, closed="left"), str, None],
        "num_splines": [Interval(Integral, 2, None, closed="left"), None],
        "edges": [None, Container],
        "degree": [Interval(Integral, 0, None, closed="left")],
        "knots": [str],
        "extrapolation": [str],
    }

    def __init__(
        self,
        feature=None,
        penalty=1,
        by=None,
        num_splines=20,
        constraints=None,
        edges=None,
        degree=3,
        knots="uniform",
        extrapolation="linear",
    ):
        """

        Examples
        --------
        >>> Spline(0)
        Spline(feature=0)
        >>> Spline(1, penalty=0.1, by=2)
        Spline(by=2, feature=1, penalty=0.1)

        >>> spline = Spline(0, num_splines=3, degree=1, extrapolation="linear")
        >>> X = np.linspace(0, 1, num=9).reshape(-1, 1)
        >>> spline.fit(X[:6, :])
        Spline(degree=1, feature=0, num_splines=3)
        >>> spline.transform(X[:6, :])
        array([[1. , 0. , 0. ],
               [0.6, 0.4, 0. ],
               [0.2, 0.8, 0. ],
               [0. , 0.8, 0.2],
               [0. , 0.4, 0.6],
               [0. , 0. , 1. ]])
        >>> spline.transform(X)
        array([[ 1. ,  0. ,  0. ],
               [ 0.6,  0.4,  0. ],
               [ 0.2,  0.8,  0. ],
               [ 0. ,  0.8,  0.2],
               [ 0. ,  0.4,  0.6],
               [ 0. ,  0. ,  1. ],
               [ 0. , -0.4,  1.4],
               [ 0. , -0.8,  1.8],
               [ 0. , -1.2,  2.2]])
        """
        self.feature = feature
        self.penalty = penalty
        self.by = by
        self.num_splines = num_splines
        self.constraints = constraints
        self.edges = edges
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation

    def _validate_params(self, X):
        # Validate using BaseEsimator._validate_params, which in turn calls
        # sklearn.utils._param_validation.validate_parameter_constraints
        # using the `_parameter_constraints` attributed defined on the class.
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/base.py#L573
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/utils/_param_validation.py#L28
        super()._validate_params()

        if self.by == self.feature:
            raise ValueError(f"Parameter {self.by=} cannot be equal to {self.feature=}")

        self.infer_feature_variable(variable_name="feature", X=X)
        self.infer_feature_variable(variable_name="by", X=X)

    @property
    def num_coefficients(self):
        return self.num_splines

    def penalty_matrix(self):
        super()._validate_params()  # Validate 'penalty' and 'num_coefficients'
        matrix = second_order_finite_difference(self.num_coefficients, periodic=(self.extrapolation == "periodic"))

        # Set all-zero rows to zero
        # all_zero_rows = np.all(np.isclose(matrix, 0), axis=1)
        # matrix = matrix[~all_zero_rows, :]

        matrix = np.sqrt(self.penalty) * matrix

        # Add the sum-to-zero penalty
        # matrix = np.vstack((matrix, np.ones(self.num_coefficients) * np.sqrt(self.L2_penalty)))

        return matrix

    def fit(self, X):
        """Fit to data.

        Parameters
        ----------
        X : np.ndarray
            An ndarray with 2 dimensions of shape (n_samples, n_features).

        Returns
        -------
        X : np.ndarray
            An ndarray for the term.

        Examples
        --------
        >>> spline = Spline(0, num_splines=3, degree=0)
        >>> X = np.linspace(0, 1, num=9).reshape(-1, 1)
        >>> spline.fit_transform(X)
        array([[1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [0., 0., 1.],
               [0., 0., 1.]])
        >>> X = np.vstack((np.linspace(0, 1, num=12), np.arange(12))).T
        >>> Spline(0, num_splines=3, degree=1).fit_transform(X).round(1)
        array([[1. , 0. , 0. ],
               [0.8, 0.2, 0. ],
               [0.6, 0.4, 0. ],
               [0.5, 0.5, 0. ],
               [0.3, 0.7, 0. ],
               [0.1, 0.9, 0. ],
               [0. , 0.9, 0.1],
               [0. , 0.7, 0.3],
               [0. , 0.5, 0.5],
               [0. , 0.4, 0.6],
               [0. , 0.2, 0.8],
               [0. , 0. , 1. ]])

        """
        self._validate_params(X)  # Get feature names, validate parameters
        X = check_array(X, estimator=self, input_name="X")  # Conver to array
        num_samples, num_features = X.shape

        X_feature = X[:, self.feature_]

        # Solve this equation for the number of knots
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/preprocessing/_polynomial.py#L470
        n_knots = self.num_splines + 1 - self.degree * (self.extrapolation != "periodic")

        self.spline_transformer_ = SplineTransformer(
            n_knots=n_knots,
            degree=self.degree,
            knots=self.knots,
            extrapolation=self.extrapolation,
            include_bias=True,
            order="C",
        )

        # Fit to data within the edges
        if self.edges is not None:
            low, high = self.edges
            mask = (X_feature >= low) & (X_feature <= high)
            X_feature_masked = X_feature[mask].reshape(-1, 1)
        else:
            X_feature_masked = X_feature.reshape(-1, 1)

        spline_basis_matrix = self.spline_transformer_.fit_transform(X_feature_masked)
        self.means_ = np.mean(spline_basis_matrix, axis=0)

        return self

    def transform(self, X):
        """transform the term.

        Parameters
        ----------
        X : np.ndarray
            An ndarray with 2 dimensions of shape (n_samples, n_features).

        Returns
        -------
        X : np.ndarray
            An ndarray for the term.

        Examples
        --------
        >>> spline = Spline(0, num_splines=3, degree=0)
        >>> X = np.linspace(0, 1, num=9).reshape(-1, 1)
        """
        self._validate_params(X)  # Get feature names, validate parameters
        X = check_array(X, estimator=self, input_name="X")  # Conver to array
        num_samples, num_features = X.shape

        X_feature = X[:, self.feature_]

        spline_basis_matrix = self.spline_transformer_.transform(X_feature.reshape(-1, 1))

        assert spline_basis_matrix.shape == (num_samples, self.num_splines)

        # Set the 'by' variable
        if self.by is not None:
            # Multiply the spline basis by the desired column
            spline_basis_matrix *= X[:, self.by_][:, np.newaxis]

        assert spline_basis_matrix.shape == (num_samples, self.num_coefficients)

        # Center to sum over data is zero

        spline_basis_matrix = spline_basis_matrix - self.means_

        return spline_basis_matrix

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Tensor(TransformerMixin, Term, BaseEstimator):
    name = "tensor"

    def __init__(self, splines, by=None):
        """


        Parameters
        ----------
        splines : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        Examples
        --------
        >>> tensor = Tensor([Spline(0), Spline(1)])
        >>> for spline in tensor:
        ...     print(spline)
        Spline(feature=0)
        Spline(feature=1)

        """
        self.splines = TermList(splines)
        self.by = by

    def __iter__(self):
        return iter(self.splines)

    def _validate_params(self, num_features):
        self.splines = TermList(self.splines)
        for spline in self.splines:
            if not isinstance(spline, Spline):
                raise TypeError(f"Only Splines can be used in a Tensor, found: {spline}")
            spline._validate_params(num_features)

    @property
    def num_coefficients(self):
        return np.product([spline.num_coefficients for spline in self.splines])

    def _build_marginal_penalties(self, i):
        """

        Examples
        --------
        >>> spline1 = Spline(0, num_splines=3, penalty=1)
        >>> spline2 = Spline(1, num_splines=4, penalty=1)
        >>> Tensor([spline1, spline2])._build_marginal_penalties(0).astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0, -2,  0,  0,  0,  1,  0,  0,  0],
               [ 0,  1,  0,  0,  0, -2,  0,  0,  0,  1,  0,  0],
               [ 0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  1,  0],
               [ 0,  0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        >>> Tensor([spline1, spline2])._build_marginal_penalties(1).astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        """

        # i = 0 -> sp.sparse.kron(term.build_penalties(), sparse.eye, sparse.eye)
        # i = 1 -> sp.sparse.kron(sparse.eye, term.build_penalties(), sparse.eye)
        # i = 1 -> sp.sparse.kron(sparse.eye, sparse.eye, term.build_penalties())

        penalty_matrices = [
            (spline.penalty_matrix() if i == j else np.eye(spline.num_coefficients))
            for j, spline in enumerate(self.splines)
        ]
        return functools.reduce(sp.linalg.kron, penalty_matrices)

    def penalty_matrix(self):
        """
        builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]

        Parameters
        ----------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form

        Examples
        --------
        The coefficients are imagined to be structured as
        [[b_11, b_12, b_13, b14],
         [b_21, b_22, b_23, b24],
         [b_31, b_32, b_33, b34]]
        and .ravel()'ed into a vector of
        [b_11, b_12, b_13, b_14, b_21, b_22, ...]
        The example below shows a penalty matrix:

        >>> spline1 = Spline(0, num_splines=3, penalty=1)
        >>> spline2 = Spline(1, num_splines=4, penalty=1)
        >>> Tensor([spline1, spline2]).penalty_matrix().astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0, -2,  0,  0,  0,  1,  0,  0,  0],
               [ 0,  1,  0,  0,  1, -4,  1,  0,  0,  1,  0,  0],
               [ 0,  0,  1,  0,  0,  1, -4,  1,  0,  0,  1,  0],
               [ 0,  0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        """
        marginal_penalty_matrices = [self._build_marginal_penalties(i) for i, _ in enumerate(self.splines)]
        return functools.reduce(np.add, marginal_penalty_matrices)

    def fit(self, X):
        X = check_array(X, estimator=self, input_name="X")
        num_samples, num_features = X.shape
        self._validate_params(num_features)

        for spline in self.splines:
            spline.fit(X)

        return self

    def transform(self, X):
        fit_matrices = [spline.transform(X) for spline in self.splines]
        spline_basis = functools.reduce(tensor_product, fit_matrices)

        # if self.by is not None:
        #    spline_basis *= X[:, self.by][:, np.newaxis]

        return spline_basis

    def get_params(self, deep=True):
        if not deep:
            return {"splines": copy.deepcopy(self.splines)}
            return super().get_params()

        out = dict()
        for i, spline in enumerate(self.splines):
            if hasattr(spline, "get_params") and not isinstance(spline, type):
                deep_items = spline.get_params().items()
                key = str(i)
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = spline
        return out

    def __sklearn_clone__(self):
        """

        Examples
        --------
        >>> from sklearn.base import clone
        >>> tensor = Tensor([Spline(0), Spline(1)])
        >>> cloned_tensor = clone(tensor)
        >>> cloned_tensor
        Tensor(TermList([Spline(feature=0), Spline(feature=1)]))

        Mutating the original will not change the clone:

        >>> tensor.set_params(**{'0__feature': 99})
        Tensor(TermList([Spline(feature=99), Spline(feature=1)]))
        >>> cloned_tensor
        Tensor(TermList([Spline(feature=0), Spline(feature=1)]))
        """

        return type(self)([copy.deepcopy(spline) for spline in self.splines])

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.

        Examples
        --------

        Setting with a shallow copy:

        >>> tensor = Tensor([Spline(0), Spline(1)])
        >>> params_shallow = tensor.get_params(deep=False)
        >>> params_shallow
        {'splines': TermList([Spline(feature=0), Spline(feature=1)])}
        >>> params_changed = {'splines': [Spline(feature=0), Spline(feature=99)]}
        >>> new_tensor = tensor.set_params(**params_changed)
        >>> new_tensor
        Tensor(TermList([Spline(feature=0), Spline(feature=99)]))
        >>> tensor
        Tensor(TermList([Spline(feature=0), Spline(feature=99)]))

        Setting with a deep copy:

        >>> terms = TermList([Linear(0), Intercept()])
        >>> params_deep = terms.get_params(deep=True)
        >>> params_deep
        {'0__by': None, '0__feature': 0, '0__penalty': 1, '0': Linear(feature=0), '1': Intercept()}
        >>> params_new = {'0__by': None, '0__feature': 2, '0__penalty': 2, '0': Linear(feature=0), '1': Intercept()}
        >>> new_terms = terms.set_params(**params_new)
        >>> new_terms
        TermList([Linear(feature=2, penalty=2), Intercept()])
        >>> terms
        TermList([Linear(feature=2, penalty=2), Intercept()])


        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        # If parameters are of the form
        # {'splines': [Linear(feature=0), Intercept()]}
        if "splines" in params.keys():
            self.splines = TermList(params["splines"])
            return self

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            # Split the key, which indicates which Term in the TermList to
            # update
            # '0__feature'.partition("__") -> ('0', '__', 'feature')
            key, delim, sub_key = key.partition("__")

            # Got an item like (('0', '__', 'feature'), 1)
            # This is an updated of a Term parameter, so we store it
            if delim:
                nested_params[key][sub_key] = value

            # Got an item like (('0', '', ''), Linear(feature=0))
            # This is an update of a Term instance, so we update immediately
            else:
                self.splines[int(key)] = value

        # Update all term paramters
        for key, value in nested_params.items():
            self.splines[int(key)].set_params(**value)

        return self

    def __repr__(self):
        classname = type(self).__name__
        return f"{classname}({self.splines.__repr__()})"


# =============================================================================
# TERMLIST
# =============================================================================


class TermList(UserList, BaseEstimator):
    def __init__(
        self,
        data=(),
    ):
        """A list of unique terms.


        Parameters
        ----------
        data : TYPE, optional
            DESCRIPTION. The default is ().
        / : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        Examples
        --------
        >>> terms = TermList([Linear(0), Intercept()])
        >>> terms
        TermList([Linear(feature=0), Intercept()])
        >>> TermList(Intercept())
        TermList([Intercept()])
        >>> print(terms)
        Linear(feature=0) + Intercept()

        Adding terms also produces a TermList.

        >>> Intercept() + Linear(0)
        TermList([Intercept(), Linear(feature=0)])
        >>> Intercept() + Linear(0) + Linear(1)
        TermList([Intercept(), Linear(feature=0), Linear(feature=1)])

        The TermList behaves very much like a list

        >>> terms = TermList([Linear(0), Intercept()])
        >>> terms += Spline(1)
        >>> terms
        TermList([Linear(feature=0), Intercept(), Spline(feature=1)])
        >>> terms[:2]
        TermList([Linear(feature=0), Intercept()])
        >>> terms[0] = 1
        Traceback (most recent call last):
         ...
        TypeError: Only terms can be added to TermList, not 1

        Calling .transform() on a TermList will compile each Term in turn.

        >>> X = np.tile(np.arange(10), reps=(2, 1)).T
        >>> terms = Intercept() + Linear(0) + Spline(1, degree=0, num_splines=2)
        >>> terms.fit_transform(X)
        array([[1., 0., 1., 0.],
               [1., 1., 1., 0.],
               [1., 2., 1., 0.],
               [1., 3., 1., 0.],
               [1., 4., 1., 0.],
               [1., 5., 0., 1.],
               [1., 6., 0., 1.],
               [1., 7., 0., 1.],
               [1., 8., 0., 1.],
               [1., 9., 0., 1.]])
        """

        super().__init__()

        if isinstance(data, Term):
            self.append(data)
        else:
            for item in data:
                self.append(item)

    def append(self, item, /):
        if not isinstance(item, Term):
            raise TypeError(f"Only terms can be added to TermList, not {item} of type {type(item)}")

        for term in self:
            if item.is_redudance_with_respect_to(term):
                msg = f"Cannot add {item} because it is redundance wrt {term}"
                raise ValueError(msg)

        super().append(item)

    def __iadd__(self, other):
        """Implement self += other."""
        for item in other:
            self.append(item)

        return self

    def __setitem__(self, key, value):
        """Set self[key] to value."""
        if not isinstance(value, Term):
            raise TypeError(f"Only terms can be added to TermList, not {value}")

        super().__setitem__(key, value)

    def __mul__(self, n, /):
        raise NotImplementedError

    def __repr__(self):
        classname = type(self).__name__
        return f"{classname}({super().__repr__()})"

    def __str__(self):
        return " + ".join(repr(term) for term in self)

    def fit(self, X):
        return np.hstack([term.fit(X) for term in self])

    def transform(self, X):
        return np.hstack([term.transform(X) for term in self])

    def fit_transform(self, X):
        return np.hstack([term.fit_transform(X) for term in self])

    @property
    def coef_(self):
        if not all(hasattr(term, "coef_") for term in self):
            raise AttributeError(f"{type(self)} object has no attribute 'coef_'")
        else:
            return np.hstack(tuple(term.coef_ for term in self))

    def penalty_matrix(self):
        penalty_matrices = [term.penalty_matrix() for term in self]
        return sp.linalg.block_diag(*penalty_matrices)

    def __sklearn_clone__(self):
        return type(self)([copy.deepcopy(term) for term in self])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.

        Examples
        --------
        >>> terms = TermList([Linear(0), Intercept()])
        >>> params_shallow = terms.get_params(deep=False)
        >>> params_shallow
        {'data': [Linear(feature=0), Intercept()]}
        >>> TermList(**params_shallow)
        TermList([Linear(feature=0), Intercept()])
        >>> params_deep = terms.get_params(deep=True)
        >>> params_deep
        {'0__by': None, '0__feature': 0, '0__penalty': 1, '0': Linear(feature=0), '1': Intercept()}

        """
        if not deep:
            return super().get_params()

        out = dict()
        for i, spline in enumerate(self):
            if hasattr(spline, "get_params") and not isinstance(spline, type):
                deep_items = spline.get_params().items()
                key = str(i)
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = spline
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.

        Examples
        --------

        Setting with a shallow copy:

        >>> terms = TermList([Linear(0), Intercept()])
        >>> params_shallow = terms.get_params(deep=False)
        >>> params_shallow
        {'data': [Linear(feature=0), Intercept()]}
        >>> params_changed = {'data': [Linear(feature=1, by=2), Intercept()]}
        >>> new_terms = terms.set_params(**params_changed)
        >>> new_terms
        TermList([Linear(by=2, feature=1), Intercept()])
        >>> terms
        TermList([Linear(by=2, feature=1), Intercept()])

        Setting with a deep copy:

        >>> terms = TermList([Linear(0), Intercept()])
        >>> params_deep = terms.get_params(deep=True)
        >>> params_deep
        {'0__by': None, '0__feature': 0, '0__penalty': 1, '0': Linear(feature=0), '1': Intercept()}
        >>> params_new = {'0__by': None, '0__feature': 2, '0__penalty': 2, '0': Linear(feature=0), '1': Intercept()}
        >>> new_terms = terms.set_params(**params_new)
        >>> new_terms
        TermList([Linear(feature=2, penalty=2), Intercept()])
        >>> terms
        TermList([Linear(feature=2, penalty=2), Intercept()])

        A TermList may be cloned:

        >>> from sklearn.base import clone
        >>> terms = TermList([Linear(0, by=2, penalty=9), Intercept()])
        >>> clone(terms)
        TermList([Linear(by=2, feature=0, penalty=9), Intercept()])


        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        # If parameters are of the form
        # {'data': [Linear(feature=0), Intercept()]}
        if "data" in params.keys():
            self.data = params["data"]
            return self

        nested_params = defaultdict(dict)  # grouped by prefix

        for key, value in params.items():
            # Split the key, which indicates which Term in the TermList to
            # update
            # '0__feature'.partition("__") -> ('0', '__', 'feature')
            key, delim, sub_key = key.partition("__")

            # Got an item like (('0', '__', 'feature'), 1)
            # This is an updated of a Term parameter, so we store it
            if delim:
                nested_params[key][sub_key] = value

            # Got an item like (('0', '', ''), Linear(feature=0))
            # This is an update of a Term instance, so we update immediately
            else:
                self[int(key)] = value

        # Update all term paramters
        for key, value in nested_params.items():
            self[int(key)].set_params(**value)

        return self


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])
