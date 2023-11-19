#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v05/undervisningsmateriale/hele.pdf

"""
import copy
import functools
from abc import ABC, abstractmethod
from collections import UserList, defaultdict
from collections.abc import Container, Iterable
from numbers import Integral, Real

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import _get_feature_names, check_is_fitted

from generalized_additive_models.penalties import second_order_finite_difference
from generalized_additive_models.splinetransformer import SplineTransformer
from generalized_additive_models.utils import tensor_product


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

    def _get_column(self, X, selector="feature"):
        # A Tensor can select several columns, so we recursively do that
        if isinstance(self, Tensor) and selector == "feature":
            return np.hstack([spline._get_column(X, selector=selector) for spline in self])

        selector = getattr(self, selector + "_")

        if hasattr(X, "iloc"):
            return X.iloc[:, selector].values.reshape(-1, 1)
        return X[:, selector].reshape(-1, 1)

    def _infer_feature_variable(self, *, variable_name, X):
        """Infer feature variable such as `feature` or `by` and set `feature_` or `by_`.

        Examples
        --------
        >>> import pandas as pd
        >>> X = np.arange(12).reshape(-1, 4)
        >>> df = pd.DataFrame(X, columns=list('abcd'))
        >>> term = Spline('b')
        >>> term._infer_feature_variable(variable_name='feature', X=df)
        >>> term.feature_
        1


        """
        # Variable name is typically 'penalty' or 'by'
        num_samples, num_features = X.shape
        variable_content = getattr(self, variable_name)
        variable_to_set = f"{variable_name}_"

        # Set the self.feature_ variable
        if isinstance(variable_content, str):
            feature_names = _get_feature_names(X)  # None or np.array
            feature_names = [] if feature_names is None else list(feature_names)
            if (num_features > 1) and (variable_content not in feature_names):
                msg = f"Feature in {self} does not match feature names in the data: {feature_names}."
                raise ValueError(msg)

            # A single column was passed, assume it's the one to transform
            # TODO: Disallow this? Might be too implicit
            elif num_features == 1 and hasattr(self, variable_to_set):
                setattr(self, variable_to_set, 0)

            else:
                setattr(self, variable_to_set, feature_names.index(variable_content))
        elif isinstance(variable_content, Integral):
            if (num_features > 1) and variable_content not in range(num_features):
                raise ValueError(f"Parameter `{variable_name}` must be in range [0, {num_features-1}].")

            # A single column was passed, assume it's the one to transform
            elif num_features == 1 and hasattr(self, variable_to_set):
                setattr(self, variable_to_set, 0)
            else:
                # Copy it over
                setattr(self, variable_to_set, variable_content)

    def is_redundant_with_respect_to(self, other):
        """Check if a term is redundant with respect to another.

        Examples
        --------
        >>> Spline(0).is_redundant_with_respect_to(Spline(0))
        True
        >>> Spline(0).is_redundant_with_respect_to(Spline(0, by=1))
        False
        >>> Intercept().is_redundant_with_respect_to(Intercept())
        True
        >>> Linear(0).is_redundant_with_respect_to(Linear(1))
        False
        >>> Linear(0).is_redundant_with_respect_to(Linear(0))
        True
        >>> te1 = Tensor([Spline(0), Spline(1)])
        >>> te2 = Tensor([Spline(1), Spline(0)])
        >>> te1.is_redundant_with_respect_to(te2)
        True
        >>> te2 = Tensor([Spline(1), Spline(0, by=1)])
        >>> te1.is_redundant_with_respect_to(te2)
        False

        """
        # Check if equal type
        equal_type = type(self) is type(other)
        if not equal_type:
            return False

        # Check for Spline/Linear/Intercept, etc
        if isinstance(self, (Intercept, Linear, Spline, Categorical)):
            return frozenset([self.feature, self.by]) == frozenset([other.feature, other.by])
        # Check for Tensor
        elif isinstance(self, Tensor):
            self_vars = frozenset([frozenset([term.feature, term.by]) for term in self] + [self.by])
            other_vars = frozenset([frozenset([term.feature, term.by]) for term in other] + [other.by])
            return self_vars == other_vars
        else:
            raise TypeError(f"Cannot compare {self} and {other}")

    def __eq__(self, other):
        # Two terms are equal iff their parameters are equal
        if type(self) != type(other):
            return False

        return self.get_params() == other.get_params()

    def __add__(self, other):
        if isinstance(other, Integral) and other == 0:
            return self
        return TermList(self) + TermList(other)

    def __radd__(self, other):
        return self.__add__(other)

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
    """An intercept term.

    Examples
    --------
    >>> intercept = Intercept()
    >>> intercept.num_coefficients
    1
    >>> import numpy as np
    >>> X = np.random.randn(5, 3)
    >>> intercept.fit_transform(X)
    array([[1.],
           [1.],
           [1.],
           [1.],
           [1.]])

    Intercepts have no penalty:

    >>> intercept.penalty_matrix()
    array([[0.]])

    Terms can yield themselves once, like so:

    >>> list(intercept)
    [Intercept()]
    """

    name = "intercept"  #: Name of the term.
    feature = None  #: Feature name, not used in Intercept.
    by = None  #: Multiplicative feature, not used in Intercept.
    _lower_bound = np.array([-np.inf])
    _upper_bound = np.array([np.inf])

    def __init__(self):
        """Initialize an Intercept."""
        pass

    @property
    def num_coefficients(self):
        """Number of coefficients for the term."""
        return 1

    def penalty_matrix(self):
        """Return the penalty matrix for the term."""
        return np.array([[0.0]])

    def fit(self, X):
        """Fit to data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset of shape (num_samples, num_features).

        """
        self.feature_ = None  # Add underscore parameter to signal fitted
        return self

    def transform(self, X):
        """Transform the input.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset of shape (num_samples, num_features).

        Returns
        -------
        X : np.ndarray
            An ndarray for the term.

        Examples
        --------
        >>> intercept = Intercept()
        >>> X = np.eye(3)
        >>> intercept.fit_transform(X)
        array([[1.],
               [1.],
               [1.]])

        """
        check_is_fitted(self)
        n_samples, n_features = X.shape
        return np.ones(n_samples).reshape(-1, 1)


class Linear(TransformerMixin, Term, BaseEstimator):
    """A linear term.

    Examples
    --------
    >>> linear = Linear(feature=0)
    >>> linear.num_coefficients
    1

    Fitting and transforming extracts the relevant column:

    >>> import numpy as np
    >>> X = np.arange(24).reshape(8, 3)
    >>> linear.fit_transform(X)
    array([[ 0],
           [ 3],
           [ 6],
           [ 9],
           [12],
           [15],
           [18],
           [21]])

    Linear terms have a standard quadratic (l2) penalty:

    >>> linear.penalty_matrix()
    array([[1.]])

    The square root of the penalty matrix is returned, so we must square it
    to get back the given penalty:

    >>> Linear(feature=0, penalty=5).penalty_matrix()**2
    array([[5.]])
    """

    name = "linear"  #: Name of the term.

    _parameter_constraints = {
        "feature": [Interval(Integral, 0, None, closed="left"), str, None],
        "penalty": [Interval(Real, 0, None, closed="left")],
        "by": [Interval(Integral, 0, None, closed="left"), str, None],
        "constraint": [StrOptions({"increasing", "decreasing"}), None],
    }

    def __init__(self, feature=None, *, penalty=1, by=None, constraint=None):
        """Initialize a Linear term.

        Parameters
        ----------
        feature : str or int, optional
            The feature name or index associated with the term.
            The default is None.
        penalty : float, optional
            Regularization penalty. The default is 1.
        by : str or int, optional
            The feature name or index associated with a multiplicative term.
            The default is None.
        constraint : str or None, optional
            Either None, or `increasing` or `decreasing`.
            The default is None.

        Examples
        --------
        >>> linear_term = Linear(0, penalty=2)
        >>> linear_term
        Linear(feature=0, penalty=2)
        >>> Linear(0, penalty=2) == Linear(0, penalty=2)
        True
        >>> Linear(0, penalty=2) == Linear(0, penalty=3)
        False
        >>> Linear(0, penalty=2).is_redundant_with_respect_to(Linear(0, penalty=3))
        True
        """
        self.feature = feature
        self.penalty = penalty
        self.by = by
        self.constraint = constraint

    def _validate_params(self, X):
        # Validate using BaseEsimator._validate_params, which in turn calls
        # sklearn.utils._param_validation.validate_parameter_constraints
        # using the `_parameter_constraints` attributed defined on the class.
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/base.py#L573
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/utils/_param_validation.py#L28
        super()._validate_params()

        if self.feature is None:
            raise ValueError(f"Feature cannot be None in term: {self}")

        if self.by == self.feature:
            raise ValueError(f"Parameter {self.by=} cannot be equal to {self.feature=}")

        self._infer_feature_variable(variable_name="feature", X=X)
        self._infer_feature_variable(variable_name="by", X=X)

    @property
    def num_coefficients(self):
        """Number of coefficients for the term."""
        return 1

    def penalty_matrix(self):
        """Return the penalty matrix for the term."""
        super()._validate_params()  # Validate the 'penalty' parameter
        return np.sqrt(self.penalty) * np.array([[1.0]])

    def fit(self, X):
        """Fit to data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset of shape (num_samples, num_features).

        """
        self._validate_params(X)

        basis_matrix = self._get_column(X, selector="feature")

        # Must apply 'by' before computing mean, since we want symmetry to hold:
        # x_1 * x_2 - mean(x_1 * x_2) = x_2 * x_1 - mean(x_2 * x_1)
        # If the computed means first, symmetry would not hold:
        # (x_1 - mean(x_1)) * x_2 != (x_2 - mean(x_2)) * x_1

        if self.by is not None:
            basis_matrix = basis_matrix * self._get_column(X, selector="by")

        if self.constraint in ("concave", "decreasing"):
            basis_matrix = -basis_matrix

        # Set up bounds
        self._lower_bound = np.array([-np.inf])
        self._upper_bound = np.array([np.inf])
        if self.constraint == "increasing":
            self._lower_bound = np.array([0])
        elif self.constraint == "decreasing":
            self._upper_bound = np.array([0])
        elif self.constraint is None:
            pass
        else:
            raise ValueError(f"Invalid constraint value: {self.constraint}")

        return self

    def transform(self, X):
        """Transform the input.

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
        >>> linear.fit_transform(X)
        array([[0.],
               [1.],
               [0.]])

        """
        check_is_fitted(self)
        self._validate_params(X)

        basis_matrix = self._get_column(X, selector="feature")

        if self.by is not None:
            basis_matrix = basis_matrix * self._get_column(X, selector="by")

        return basis_matrix


class Spline(TransformerMixin, Term, BaseEstimator):
    """A Spline term.

    Parameters
    ----------
    feature : int or str, optional
        The column index of the feature, or the name of the feature if the
        data set is a pandas DataFrame. The default is None.
    penalty : float, optional
        A penalty term that penalizes the second derivative of the spline.
        If set high, the spline becomes linear (no second derivative).
        If set low, the spline becomes very wiggly and tends to overfit.
        The default is 1.
    by : int or str, optional
        An interaction effect with a numerical feature. The spline

        > Spline("age", by="income")

        models the multiplicative interaction :math:`\text{income} \times f(\text{age})`,
        meaning that the target is modeled as a smooth function of age, times
        a linear function of income. The default is None.
    num_splines : int, optional
        The number of spline basis functions. The default is 20.
    constraint : TYPE, optional
        A constraint for the spline. Must be one of
        {'increasing-concave', 'convex', 'decreasing-concave', 'increasing',
        'concave', 'decreasing', 'decreasing-convex', 'increasing-convex'}
        or None. The constraints do not hold for all extrapolations.
        The default is None.
    edges : tuple, optional
        A tuple with edges (low, high). For instance, to model a 24 hour
        periodic phenomenon, we could use

        > Spline("time", edges=(0, 24), extrapolation="periodic")

        The default is None, meaning that edges are inferred from the data.
    degree : int, optional
        The spline degree. Degree 0 are box function, degree 1 are hat
        functions (also called tent functions), degree 2 are quadratic and
        degree 3 are cubic, and so forth.
    knots : str, optional
        Where to place the knots, must be in {'quantile', 'uniform'}.
    extrapolation : str, optional
        Must be one of {'continue', 'linear', 'error', 'constant', 'periodic'}.

    Returns
    -------
    None.

    Examples
    --------
    >>> spline = Spline(feature=0, num_splines=8)
    >>> spline.num_coefficients
    8

    Fitting and transforming creates a spline basis. The basis is given a
    sum-to-zero constraint over the data it is fitted on.

    >>> import numpy as np
    >>> X = np.arange(27).reshape(9, 3)
    >>> Spline(0, num_splines=3, degree=0).fit_transform(X).round(2)
    array([[ 0.67, -0.33, -0.33],
           [ 0.67, -0.33, -0.33],
           [ 0.67, -0.33, -0.33],
           [-0.33,  0.67, -0.33],
           [-0.33,  0.67, -0.33],
           [-0.33,  0.67, -0.33],
           [-0.33, -0.33,  0.67],
           [-0.33, -0.33,  0.67],
           [-0.33, -0.33,  0.67]])

    To recover the un-centered splines, we can add by the means learned
    during fitting:

    >>> spline = Spline(0, num_splines=3, degree=0)
    >>> spline = spline.fit(X)
    >>> spline.transform(X) + spline.means_
    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 1.]])

    Splines are given a penalty over the smoothness as measured by the second
    derivative. The second deriative is given by [1, -2, 1]:

    >>> spline.penalty_matrix()
    array([[ 0.,  0.,  0.],
           [ 1., -2.,  1.],
           [ 0.,  0.,  0.]])

    The structure is more easily seen on a Spline with `num_splines` set higher.

    >>> Spline(0, num_splines=6).penalty_matrix()
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 1., -2.,  1.,  0.,  0.,  0.],
           [ 0.,  1., -2.,  1.,  0.,  0.],
           [ 0.,  0.,  1., -2.,  1.,  0.],
           [ 0.,  0.,  0.,  1., -2.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]])

    The level of penalization is given by the `penalty` parameter:

    >>> Spline(0, num_splines=6, penalty=9).penalty_matrix()
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 3., -6.,  3.,  0.,  0.,  0.],
           [ 0.,  3., -6.,  3.,  0.,  0.],
           [ 0.,  0.,  3., -6.,  3.,  0.],
           [ 0.,  0.,  0.,  3., -6.,  3.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]])

    Linear functions are in the null space of the penalty:

    >>> P = Spline(0, num_splines=6).penalty_matrix()
    >>> np.linalg.norm(P @ np.arange(6))**2
    0.0
    """

    name = "spline"  #: Name of the term.

    _parameter_constraints = {
        "feature": [Interval(Integral, 0, None, closed="left"), str, None],
        "penalty": [Interval(Real, 0, None, closed="left")],
        "by": [Interval(Integral, 0, None, closed="left"), str, None],
        "num_splines": [Interval(Integral, 2, None, closed="left"), None],
        "constraint": [
            StrOptions(
                {
                    "increasing",
                    "decreasing",
                    "convex",
                    "concave",
                    "increasing-convex",
                    "increasing-concave",
                    "decreasing-convex",
                    "decreasing-concave",
                }
            ),
            None,
        ],
        "edges": [None, Container],
        "degree": [Interval(Integral, 0, None, closed="left")],
        "knots": [StrOptions({"uniform", "quantile"})],
        "extrapolation": [StrOptions({"error", "constant", "linear", "continue", "periodic"})],
    }

    def __init__(
        self,
        feature=None,
        *,
        penalty=1,
        by=None,
        num_splines=20,
        constraint=None,
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
        >>> spline.transform(X[:6, :]) + spline.means_
        array([[1. , 0. , 0. ],
               [0.6, 0.4, 0. ],
               [0.2, 0.8, 0. ],
               [0. , 0.8, 0.2],
               [0. , 0.4, 0.6],
               [0. , 0. , 1. ]])
        >>> spline.transform(X) + spline.means_
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
        self.constraint = constraint
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

        if self.feature is None:
            raise ValueError(f"Feature cannot be None in term: {self}")

        if self.by == self.feature:
            raise ValueError(f"Parameter {self.by=} cannot be equal to {self.feature=}")

        self._infer_feature_variable(variable_name="feature", X=X)
        self._infer_feature_variable(variable_name="by", X=X)

    @property
    def num_coefficients(self):
        """Number of coefficients for the term."""
        return self.num_splines

    def penalty_matrix(self):
        """Return the penalty matrix for the term."""
        super()._validate_params()  # Validate 'penalty' and 'num_coefficients'
        periodic = self.extrapolation == "periodic"
        matrix = second_order_finite_difference(self.num_coefficients, periodic=periodic)

        return np.sqrt(self.penalty) * matrix

    def _post_transform_basis_for_constraint(self, *, constraint, basis_matrix, basis_matrix_mirrored, X_feature):
        """Transform basis matrices to comply with constraints.

        The idea is from Meyer, see: https://arxiv.org/abs/0811.1705
        """

        _upper_bound = np.array([np.inf] * self.num_coefficients)

        if constraint is None:
            _lower_bound = np.array([-np.inf] * self.num_coefficients)
            return _lower_bound, _upper_bound, basis_matrix

        if constraint in ("increasing", "increasing-convex"):
            basis_matrix = basis_matrix

        elif constraint in ("decreasing", "decreasing-concave"):
            basis_matrix = -basis_matrix

        elif constraint == "decreasing-convex":
            basis_matrix = basis_matrix_mirrored

        elif constraint == "convex":
            basis_matrix = basis_matrix + basis_matrix_mirrored[:, ::-1]

        elif constraint == "increasing-concave":
            basis_matrix = -basis_matrix_mirrored

        elif constraint == "concave":
            basis_matrix = -basis_matrix - basis_matrix_mirrored[:, ::-1]

        _lower_bound = np.array([0] * self.num_coefficients)

        return _lower_bound, _upper_bound, basis_matrix

    def fit(self, X):
        """Fit to data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset of shape (num_samples, num_features).

        Returns
        -------
        X : np.ndarray
            An ndarray for the term.

        Examples
        --------
        >>> spline = Spline(0, num_splines=3, degree=0)
        >>> X = np.linspace(0, 1, num=9).reshape(-1, 1)
        >>> spline = spline.fit(X)
        >>> spline.transform(X) + spline.means_
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
        >>> spline = Spline(0, num_splines=3, degree=1).fit(X)
        >>> (spline.transform(X) + spline.means_).round(1)
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
        num_samples, num_features = X.shape

        X_feature = self._get_column(X, selector="feature")

        # TODO: Decrement degree if constraint, since we integrate it up again
        if self.constraint is None:
            degree_adjustment = 0
        elif self.constraint in ("increasing", "decreasing"):
            degree_adjustment = 1
        else:
            degree_adjustment = 2

        # Solve this equation for the number of knots
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/preprocessing/_polynomial.py#L470
        n_knots = self.num_splines + 1 - (self.degree - degree_adjustment) * (self.extrapolation != "periodic")

        # Set up two spline transformers
        # - The first one is fit on the data ordinarity
        # - The second one is fit on the mirrored data
        self.spline_transformer_ = SplineTransformer(
            n_knots=n_knots,
            degree=self.degree - degree_adjustment,
            knots=self.knots,
            extrapolation=self.extrapolation,
            include_bias=True,
            order="C",
        )
        self.spline_transformer_mirrored_ = clone(self.spline_transformer_)

        # Select data within the edges
        if self.edges is not None:
            low, high = self.edges
            mask = (X_feature >= low) & (X_feature <= high)
            X_feature_masked = X_feature[mask].reshape(-1, 1)
        else:
            X_feature_masked = X_feature.reshape(-1, 1)

        # Fit both spline transformers
        self.spline_transformer_.fit(X_feature_masked)
        self.spline_transformer_mirrored_.fit(-X_feature_masked)

        # If the constraint is 'increasing' or 'decreasing', antidifferentiate once
        # If the constraint is 'convex' or 'concave' or related, antidifferentiate twice
        assert len(self.spline_transformer_.bsplines_) == 1
        if degree_adjustment:
            self.spline_transformer_.bsplines_[0] = self.spline_transformer_.bsplines_[0].antiderivative(
                degree_adjustment
            )
            self.spline_transformer_mirrored_.bsplines_[0] = self.spline_transformer_mirrored_.bsplines_[
                0
            ].antiderivative(degree_adjustment)

            # Adjust the degree up again
            self.spline_transformer_.degree += degree_adjustment
            self.spline_transformer_mirrored_.degree += degree_adjustment

        # Generate basis matrix
        (
            self._lower_bound,
            self._upper_bound,
            basis_matrix,
        ) = self._post_transform_basis_for_constraint(
            constraint=self.constraint,
            basis_matrix=self.spline_transformer_.transform(X_feature_masked),
            basis_matrix_mirrored=self.spline_transformer_mirrored_.transform(-X_feature_masked),
            X_feature=X_feature,
        )

        # Center the spline basis so every column has 0 as the lowest value
        self.basis_min_value_ = np.min(basis_matrix, axis=0)
        basis_matrix = basis_matrix - self.basis_min_value_

        assert np.all(np.min(basis_matrix, axis=0) >= 0)

        # Set the 'by' variable
        if self.by is not None:
            # Multiply the spline basis by the desired column
            by_column = self._get_column(X, selector="by")
            self.min_by_ = np.min(by_column)
            basis_matrix = basis_matrix * (by_column - self.min_by_)

        self.means_ = np.mean(basis_matrix, axis=0)

        return self

    def transform(self, X):
        """Transform the input.

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
        # INFERENCE USING SHAPE-RESTRICTED REGRESSION SPLINES
        # https://arxiv.org/pdf/0811.1705.pdf

        check_is_fitted(self)
        self._validate_params(X)  # Get feature names, validate parameters
        # X = check_array(X, estimator=self, input_name="X")  # Convert to array
        num_samples, num_features = X.shape

        X_feature = self._get_column(X, selector="feature")

        (
            self._lower_bound,
            self._upper_bound,
            basis_matrix,
        ) = self._post_transform_basis_for_constraint(
            constraint=self.constraint,
            basis_matrix=self.spline_transformer_.transform(X_feature),
            basis_matrix_mirrored=self.spline_transformer_mirrored_.transform(-X_feature),
            X_feature=X_feature,
        )
        assert basis_matrix.shape == (num_samples, self.num_coefficients)

        # Apply the same centering that was done during fitting
        basis_matrix = basis_matrix - self.basis_min_value_

        # Set the 'by' variable
        if self.by is not None:
            # Multiply the spline basis by the desired column
            by_column = self._get_column(X, selector="by")
            basis_matrix = basis_matrix * (by_column - self.min_by_)

        assert basis_matrix.shape == (num_samples, self.num_coefficients)

        # Center the same was as was done during fitting
        basis_matrix = basis_matrix - self.means_
        return basis_matrix

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Tensor(TransformerMixin, Term, BaseEstimator):
    """A Tensor term.

    Examples
    --------

    A Tensor is constructed from a list of Splines or a TermList with Splines:

    >>> tensor = Tensor(splines=[Spline(0), Spline(1)])
    >>> tensor
    Tensor(TermList([Spline(feature=0), Spline(feature=1)]))
    >>> Tensor(Spline("age") + Spline("bmi"))
    Tensor(TermList([Spline(feature='age'), Spline(feature='bmi')]))

    The number of coefficients equals the product of each Spline's coefficients:

    >>> tensor = Tensor(Spline(0, num_splines=3) + Spline(1, num_splines=4))
    >>> tensor.num_coefficients
    12

    Fitting and transforming creates a spline basis like so:

    >>> X = np.array([[1, 1],
    ...               [1, 2],
    ...               [1, 3],
    ...               [2, 1],
    ...               [2, 2],
    ...               [2, 3],
    ...               [3, 1],
    ...               [3, 2],
    ...               [3, 3]])
    >>> tensor = Tensor(Spline(0, num_splines=3, degree=0) + Spline(1, num_splines=3, degree=0))
    >>> tensor.fit_transform(X) + tensor.means_
    array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

    Penalties are given to neighboring coefficients. In this case the result
    is hard to decipher, but it checks out. The first row gives the penalty for
    the first coefficient, and so forth:

    >>> tensor.penalty_matrix()
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 1., -2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0., -2.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  1., -4.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0., -2.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -2.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    Imagine a matrix of coefficients that looks like this:
    [beta_11, beta_12, beta_13]
    [beta_21, beta_22, beta_23]
    [beta_31, beta_32, beta_33]
    Beta_11 is the first coefficient when unpacked to a vector, with no penalty.
    Beta_12 is the second coefficient when unpacked to a vector, with a penalty
    relating it to beta_11 and beta_13.
    The pattern continues and e.g. beta_22 is related to four other coefficients,
    namely beta_12, beta_21, beta_23 and beta_32.

    The level of penalization is given by the `penalty` parameter for each
    Spline, and is multiplied together after taking square roots. Penalties can
    vary in each dimension:

    >>> spline1 = Spline(0, num_splines=3, degree=0, penalty=9)
    >>> spline2 = Spline(1, num_splines=3, degree=0, penalty=1)
    >>> tensor = Tensor(spline1 + spline2)
    >>> tensor.penalty_matrix()
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 1., -2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 3.,  0.,  0., -6.,  0.,  0.,  3.,  0.,  0.],
           [ 0.,  3.,  0.,  1., -8.,  1.,  0.,  3.,  0.],
           [ 0.,  0.,  3.,  0.,  0., -6.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -2.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    Linear functions of two variables are in the null space of the penalty

    >>> coefs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> P = tensor.penalty_matrix()
    >>> np.linalg.norm(P @ coefs)**2
    0.0

    """

    name = "tensor"  #: Name of the term.

    _parameter_constraints = {
        "splines": [Iterable],
        "by": [Interval(Integral, 0, None, closed="left"), str, None],
    }

    def __init__(self, splines, *, by=None):
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

    def _validate_params(self, X):
        super()._validate_params()
        num_samples, num_features = X.shape

        self.splines = TermList(self.splines)
        for spline in self.splines:
            if not isinstance(spline, (Spline, Categorical)):
                raise TypeError(f"Only Splines and Categorical can be used in Tensor, found: {spline}")
            spline._validate_params(X)

        self._infer_feature_variable(variable_name="by", X=X)

    @property
    def num_coefficients(self):
        """Number of coefficients for the term."""
        return np.prod([spline.num_coefficients for spline in self.splines])

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
        """Return the penalty matrix for the term.


        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        """Build the penaltry matrix.
        
        builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]

        Returns
        -------
        np.ndarray
            Penaltry matrix.

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
        """Fit to data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset of shape (num_samples, num_features).

        """

        self._validate_params(X)

        # Fit splines to learn individual mean values
        for spline in self.splines:
            spline.fit(X)

        # Transform, correct for individual means
        fit_matrices = [spline.transform(X) + spline.means_ for spline in self.splines]
        spline_basis = functools.reduce(tensor_product, fit_matrices)
        assert np.all(spline_basis >= 0), f"Every element in tensor basis must be >= 0 {np.min(spline_basis)}"

        # Set the 'by' variable
        if self.by is not None:
            # Multiply the spline basis by the desired column
            spline_basis = spline_basis * self._get_column(X, selector="by")

        # Learn the joint mean values
        self.means_ = np.mean(spline_basis, axis=0)

        # Set bounds
        # TODO: think about this
        self._lower_bound = np.ones(self.num_coefficients) * max(np.max(spline._lower_bound) for spline in self.splines)
        self._upper_bound = np.ones(self.num_coefficients) * min(np.min(spline._upper_bound) for spline in self.splines)
        # self._bounds = np.ones(self.num_coefficients) * max(np.max(spline._bounds) for spline in self.splines)

        return self

    def transform(self, X):
        """Transform the input."""
        check_is_fitted(self)
        self._validate_params(X)

        fit_matrices = [spline.transform(X) + spline.means_ for spline in self.splines]
        spline_basis = functools.reduce(tensor_product, fit_matrices)

        # Set the 'by' variable
        if self.by is not None:
            # Multiply the spline basis by the desired column
            spline_basis = spline_basis * self._get_column(X, selector="by")

        # Subtract the means
        spline_basis = spline_basis - self.means_

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
        {'0__by': None, '0__constraint': None, '0__feature': 0, '0__penalty': 1, '0': Linear(feature=0), '1': Intercept()}
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

        # Update all term parameters
        for key, value in nested_params.items():
            self.splines[int(key)].set_params(**value)

        return self

    def __repr__(self):
        classname = type(self).__name__
        return f"{classname}({self.splines.__repr__()})"


class Categorical(TransformerMixin, Term, BaseEstimator):
    """A Categorial term.

    Examples
    --------

    A Categorial term is just a wrapper around sklearn's OneHotEncoder.
    They are also called factor terms.

    >>> X = np.array([1, 1, 2, 1, 2, 2]).reshape(-1, 1)
    >>> categorical = Categorical(0).fit(X)
    >>> categorical.transform(X)
    array([[1., 0.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [0., 1.]])

    Or, with a DataFrame:

    >>> import pandas as pd
    >>> df = pd.DataFrame({"colors": ["red", "red", "blue", "yellow", "red"]})
    >>> categorical = Categorical("colors")
    >>> categorical.fit_transform(df)
    array([[0., 1., 0.],
           [0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])

    The number of coefficients equals the unique number of entries in the feature:

    >>> categorical.num_coefficients
    3

    Each unique entry gets a penalty, penalizing the coefficients towards zero:

    >>> categorical.penalty_matrix()
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    The categories assigned to each coefficient can be retrieved like so:

    >>> categorical.categories_
    ['blue', 'red', 'yellow']
    """

    name = "categorical"  #: Name of the term.

    _parameter_constraints = {
        "feature": [Interval(Integral, 0, None, closed="left"), str, None],
        "penalty": [Interval(Real, 0, None, closed="left")],
        "by": [Interval(Integral, 0, None, closed="left"), str, None],
    }

    def __init__(
        self,
        feature=None,
        penalty=1,
        by=None,
        handle_unknown="error",
        min_frequency=None,
        max_categories=None,
    ):
        """Create a categorial term with a given penalty.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> df = load_diabetes(as_frame=True).data.iloc[:5, :]
        >>> df.sex
        0    0.050680
        1   -0.044642
        2    0.050680
        3   -0.044642
        4   -0.044642
        Name: sex, dtype: float64
        >>> categorical_term = Categorical("sex")
        >>> categorical_term.fit_transform(df)
        array([[0., 1.],
               [1., 0.],
               [0., 1.],
               [1., 0.],
               [1., 0.]])
        >>> import pandas as pd
        >>> df = pd.DataFrame({'sex': ['M', 'F', 'M', 'F', 'F', 'Unknown']})
        >>> categorical_term = Categorical("sex")
        >>> categorical_term.fit_transform(df)
        array([[0., 1., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [0., 0., 1.]])
        >>> categorical_term.categories_
        ['F', 'M', 'Unknown']
        """
        self.feature = feature
        self.penalty = penalty
        self.by = by
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def _validate_params(self, X):
        # Validate using BaseEsimator._validate_params, which in turn calls
        # sklearn.utils._param_validation.validate_parameter_constraints
        # using the `_parameter_constraints` attributed defined on the class.
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/base.py#L573
        # https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/utils/_param_validation.py#L28
        super()._validate_params()

        if self.feature is None:
            raise ValueError(f"Feature cannot be None in term: {self}")

        if self.by == self.feature:
            raise ValueError(f"Parameter {self.by=} cannot be equal to {self.feature=}")

        self._infer_feature_variable(variable_name="feature", X=X)
        self._infer_feature_variable(variable_name="by", X=X)

    @property
    def num_coefficients(self):
        """Number of coefficients for the term."""
        return len(self.categories_)

    def penalty_matrix(self):
        """Return the penalty matrix for the term."""
        super()._validate_params()  # Validate the 'penalty' parameter
        return np.sqrt(self.penalty) * np.eye(self.num_coefficients)

    def fit(self, X):
        """Fit to data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset of shape (num_samples, num_features).

        """
        self._validate_params(X)

        self.onehotencoder_ = OneHotEncoder(
            categories="auto",
            drop=None,  # Identifiability constraints will put a coeff to zero, so keep all
            sparse_output=False,
            dtype=float,
            handle_unknown=self.handle_unknown,
            min_frequency=self.min_frequency,
            max_categories=self.max_categories,
            feature_name_combiner="concat",
        )

        # X = check_array(X, estimator=self, input_name="X")

        basis_matrix = self.onehotencoder_.fit_transform(self._get_column(X))

        # Set the 'by' variable
        if self.by is not None:
            # Multiply the spline basis by the desired column
            basis_matrix = basis_matrix * self._get_column(X, selector="by")

        self.categories_ = list(self.onehotencoder_.categories_[0])
        self.means_ = basis_matrix.mean(axis=0)

        # Set the bounds
        self._lower_bound = np.array([-np.inf for _ in range(self.num_coefficients)])
        self._upper_bound = np.array([np.inf for _ in range(self.num_coefficients)])

        return self

    def transform(self, X):
        """transform the input.

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
        >>> X = np.eye(3) * 3
        >>> linear.fit_transform(X)
        array([[0.],
               [3.],
               [0.]])
        """
        check_is_fitted(self)
        self._validate_params(X)
        num_samples, num_features = X.shape

        basis_matrix = self.onehotencoder_.transform(self._get_column(X))

        if self.by is not None:
            basis_matrix = basis_matrix * self._get_column(X, "by")

        basis_matrix = basis_matrix  # - self.means_
        return basis_matrix


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

        Calling .transform() on a TermList will transform each Term in turn.

        >>> X = np.tile(np.arange(10), reps=(2, 1)).T
        >>> terms = Intercept() + Linear(0) + Spline(1, degree=0, num_splines=2)
        >>> terms.fit_transform(X)
        array([[ 1. ,  0. ,  0.5, -0.5],
               [ 1. ,  1. ,  0.5, -0.5],
               [ 1. ,  2. ,  0.5, -0.5],
               [ 1. ,  3. ,  0.5, -0.5],
               [ 1. ,  4. ,  0.5, -0.5],
               [ 1. ,  5. , -0.5,  0.5],
               [ 1. ,  6. , -0.5,  0.5],
               [ 1. ,  7. , -0.5,  0.5],
               [ 1. ,  8. , -0.5,  0.5],
               [ 1. ,  9. , -0.5,  0.5]])
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
            if item.is_redundant_with_respect_to(term):
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

    def __add__(self, other):
        if isinstance(other, Term):
            return self.__class__(self.data + [other])
        elif isinstance(other, UserList):
            return self.__class__(self.data + other.data)
        elif isinstance(other, type(self.data)):
            return self.__class__(self.data + other)
        elif isinstance(other, Integral) and other == 0:
            return self.__class__(self.data)
        return self.__class__(self.data + list(other))

    def __mul__(self, n, /):
        raise NotImplementedError

    def __repr__(self):
        classname = type(self).__name__
        return f"{classname}({super().__repr__()})"

    def __str__(self):
        return " + ".join(repr(term) for term in self)

    def fit(self, X):
        """Fit to data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset of shape (num_samples, num_features).

        Examples
        --------
        >>> terms = TermList([Linear(0), Intercept()])
        >>> X = np.arange(6).reshape(-1, 1)
        >>> terms = terms.fit(X)
        >>> terms.num_coefficients
        2
        >>> terms.transform(X)
        array([[0., 1.],
               [1., 1.],
               [2., 1.],
               [3., 1.],
               [4., 1.],
               [5., 1.]])
        """
        for term in self:
            term.fit(X)

        self._lower_bound = np.hstack([term._lower_bound for term in self])
        self._upper_bound = np.hstack([term._upper_bound for term in self])
        return self

    @property
    def num_coefficients(self):
        """Number of coefficients for the terms."""
        return np.sum(list(term.num_coefficients for term in self))

    def transform(self, X):
        """Transform the input."""
        return np.hstack([term.transform(X) for term in self])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        return np.hstack([term.fit_transform(X) for term in self])

    @property
    def coef_(self):
        if not all(hasattr(term, "coef_") for term in self):
            raise AttributeError(f"{type(self)} object has no attribute 'coef_'")
        else:
            return np.hstack(tuple(term.coef_ for term in self))

    def penalty_matrix(self):
        """Return the penalty matrix for the terms."""
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
        {'0__by': None, '0__constraint': None, '0__feature': 0, '0__penalty': 1, '0': Linear(feature=0), '1': Intercept()}

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
        {'0__by': None, '0__constraint': None, '0__feature': 0, '0__penalty': 1, '0': Linear(feature=0), '1': Intercept()}
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

        # Update all term parameters
        for key, value in nested_params.items():
            self[int(key)].set_params(**value)

        return self


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])
