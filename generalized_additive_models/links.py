#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:36:42 2023

@author: tommy
"""

from abc import ABC, abstractmethod
from numbers import Real

import numpy as np
from scipy import special
from sklearn.base import BaseEstimator

MACHINE_EPSILON = np.finfo(float).eps
EPSILON = np.sqrt(MACHINE_EPSILON)


class Link(ABC):
    # https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
    @abstractmethod
    def link(self, mu):
        # The link function
        pass

    @abstractmethod
    def inverse_link(self, linear_prediction):
        # The inverse link function
        pass

    @abstractmethod
    def derivative(self, mu):
        # Gradient of the link function
        pass

    @abstractmethod
    def second_derivative(self, mu):
        # Gradient of the link function
        pass

    def _validate_threshold(self, threshold, argument):
        # Threshold is a number
        if not isinstance(threshold, np.ndarray):
            threshold = np.ones_like(argument, dtype=float) * threshold
        if not len(threshold) == len(argument):
            raise ValueError("Lengths of threshold in Link must match argument.")
        return threshold

    def __call__(self, *args, **kwargs):
        return self.link(*args, **kwargs)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.get_params() == other.get_params()


class Identity(Link, BaseEstimator):
    r"""Identity link: :math:`g(\mu) = \mu`"""

    name = "identity"  #: Name of the link function
    domain = (-np.inf, np.inf)  #: Domain of the link function

    def link(self, mu):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space.

        Examples
        --------
        >>> mu = np.array([-1, 0, 1])
        >>> Identity().link(mu)
        array([-1,  0,  1])
        """
        return mu

    def inverse_link(self, linear_prediction):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        return linear_prediction

    def derivative(self, mu):
        """Elementwise first derivative of the link function."""
        return np.ones_like(mu, dtype=float)

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        return np.zeros_like(mu, dtype=float)


class Logit(Link, BaseEstimator):
    r"""Logit link: :math:`g(\mu) = \log(\mu / (1 - \mu))`"""

    name = "logit"  #: Name of the link function
    domain = (0, 1)  #: Domain of the link function

    def __init__(self, low=0, high=1):
        if isinstance(low, np.ndarray) and isinstance(high, Real):
            high = np.ones_like(low, dtype=float) * high
        elif isinstance(high, np.ndarray) and isinstance(low, Real):
            low = np.ones_like(high, dtype=float) * low
        elif isinstance(low, Real) and isinstance(high, Real):
            pass
        elif isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
            pass
        else:
            raise TypeError("`low` and `high` must be array or floats.")

        self.low = low
        self.high = high
        self.domain = (self.low, self.high)

    def link(self, mu):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space.

        Examples
        --------
        >>> mu = np.array([0.1, 0.5, 0.9, 0.99, 0.999])
        >>> Logit().link(mu)
        array([-2.19722458,  0.        ,  2.19722458,  4.59511985,  6.90675478])

        """
        # x = log((-L + y)/(H - y))

        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        return np.log(mu - low) - np.log(high - mu)

    def inverse_link(self, linear_prediction):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        # expit(x) = 1 / (1 + exp(-x))

        low = self._validate_threshold(threshold=self.low, argument=linear_prediction)
        high = self._validate_threshold(threshold=self.high, argument=linear_prediction)

        return low + (high - low) * special.expit(linear_prediction)

    def derivative(self, mu):
        """Elementwise first derivative of the link function."""
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        threshold = EPSILON
        mu = np.maximum(np.minimum(mu, high - threshold), low + threshold)

        return np.exp(np.log(high - low) - np.log(high - mu) - np.log(mu - low))
        return (high - low) / ((high - mu) * (mu - low))

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        numerator = (low - high) * (high + low - 2 * mu)
        denominator = (high - mu) ** 2 * (low - mu) ** 2
        return numerator / denominator


class Log(Link, BaseEstimator):
    r"""Log link: :math:`g(\mu) = \log(\mu)`"""

    name = "log"  #: Name of the link function
    domain = (0, np.inf)  #: Domain of the link function

    def link(self, mu):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space.

        Examples
        --------
        >>> mu = np.array([0.1, 0.5, 0.9, 0.99, 0.999])
        >>> Log().link(mu)
        array([-2.30258509e+00, -6.93147181e-01, -1.05360516e-01, -1.00503359e-02,
               -1.00050033e-03])
        """
        return np.log(mu)

    def inverse_link(self, linear_prediction):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        return np.exp(linear_prediction)

    def derivative(self, mu):
        """Elementwise first derivative of the link function."""
        return 1.0 / mu

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        return -1.0 / mu**2


class SmoothLog(Link, BaseEstimator):
    r"""g(x) = (x^(a- 1) - 1) / (a - 1)

    where
    a -> 1       => logarithm
    a = 2        => linear function x - 1
    a > 1        => lower-log
    a < 1        => upper log

    """

    name = "smoothlog"  #: Name of the link function
    domain = (0, np.inf)  #: Domain of the link function

    def __init__(self, a=1.5):
        self.a = a
        assert self.a <= 2
        assert self.a != 1

    def link(self, mu):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space.

        Examples
        --------
        >>> mu = np.array([0.1, 0.5, 0.9, 0.99, 0.999])
        >>> SmoothLog(a=1.5).link(mu)
        array([-1.36754447e+00, -5.85786438e-01, -1.02633404e-01, -1.00251258e-02,
               -1.00025013e-03])
        """
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return (mu ** (a - 1) - 1) / (a - 1)

    def inverse_link(self, linear_prediction):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        # (a*y - y + 1)**(1/(a - 1))

        a = self._validate_threshold(threshold=self.a, argument=linear_prediction)

        base = linear_prediction * (a - 1) + 1
        exponent = 1 / (a - 1)
        return base**exponent

    def derivative(self, mu):
        """Elementwise first derivative of the link function."""
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return mu ** (a - 2)

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return mu ** (a - 3) * (a - 2)


class Softplus(Link, BaseEstimator):
    r"""Softplus link: :math:`g(\mu) = \log(\exp(a \mu) - 1)/a`"""

    # Using the Softplus Function to Construct Alternative Link Functions
    # in Generalized Linear Models and Beyond
    # https://arxiv.org/pdf/2111.14207.pdf

    name = "softplus"  #: Name of the link function
    domain = (0, np.inf)  #: Domain of the link function

    def __init__(self, a=1):
        r"""Initialize Softplus link.

        Parameters
        ----------
        a : float, optional
            Positive parameter indicating how closely the inverse log function
            resembles :math:`\max(0, \mu)`. Larger values of `a` makes the link
            function more closely mimic :math:`\max(0, \mu)`. The default is 1.

        """
        self.a = a
        assert a > 0

    def link(self, mu):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space.

        Examples
        --------
        >>> mu = np.array([0.01, 0.1, 1., 5., 10.])
        >>> Softplus(a=1).link(mu)
        array([-4.60016602, -2.25216846,  0.54132485,  4.99323925,  9.9999546 ])
        """
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return np.maximum(0, mu) + np.log1p(-np.exp(-a * np.abs(mu))) / a
        return np.log(np.exp(a * mu) - 1) / a

    def inverse_link(self, linear_prediction):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        # the inverse is the softplus

        a = self._validate_threshold(threshold=self.a, argument=linear_prediction)

        # return np.log(1 + np.exp(a * linear_prediction))/a
        return np.maximum(0, linear_prediction) + np.log1p(np.exp(-a * np.abs(linear_prediction))) / a

    def derivative(self, mu):
        """Elementwise first derivative of the link function."""
        a = self._validate_threshold(threshold=self.a, argument=mu)

        # If mu is 0 + epislon, then exp(mu) = 1 and we divide by zero
        threshold = EPSILON
        mu = np.maximum(mu, 0 + threshold)

        return 1 / (1 - np.exp(-a * mu))

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return -a / (4 * np.sinh(a * mu / 2) ** 2)


class CLogLogLink(Link, BaseEstimator):
    r"""g(\mu) = \log(-\log(1 - \mu))"""

    name = "cloglog"  #: Name of the link function
    domain = (0, 1)  #: Domain of the link function

    def __init__(self, low=0, high=1):
        assert type(low) == type(high)
        self.low = low
        self.high = high

    def link(self, mu, levels=1):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space."""
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        return np.log(np.log(high - low) - np.log(high - mu))

    def inverse_link(self, linear_prediction, levels=1):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        low = self._validate_threshold(threshold=self.low, argument=linear_prediction)
        high = self._validate_threshold(threshold=self.high, argument=linear_prediction)

        exponential = np.exp(-np.exp(linear_prediction))

        return high - high * exponential + low * exponential

    def derivative(self, mu, levels=1):
        """Elementwise first derivative of the link function."""
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        return 1 / ((high - mu) * (np.log(high - low) - np.log(high - mu)))

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        logterm = np.log(high - low) - np.log(high - mu)

        return (logterm - 1) / ((high - mu) ** 2 * logterm**2)


class Inverse(Link, BaseEstimator):
    r"""g(mu) = 1/mu"""

    name = "inverse"  #: Name of the link function
    domain = (0, np.inf)  #: Domain of the link function

    def link(self, mu):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space."""
        return 1.0 / mu

    def inverse_link(self, linear_prediction):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        return 1.0 / linear_prediction

    def derivative(self, mu):
        """Elementwise first derivative of the link function."""
        return -1.0 / mu**2

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        return 2.0 / mu**3


class InvSquared(Link, BaseEstimator):
    r"""g(mu) = 1/mu**2"""

    name = "inv_squared"  #: Name of the link function
    domain = (0, np.inf)  #: Domain of the link function

    def link(self, mu):
        r"""Map from the expected value :math:`\mu` to the unbounded linear space."""
        return 1.0 / mu**2

    def inverse_link(self, linear_prediction):
        r"""Map from the linear space to the expected value :math:`\mu`."""
        return 1.0 / np.sqrt(linear_prediction)

    def derivative(self, mu):
        """Elementwise first derivative of the link function."""
        return -2.0 / mu**3

    def second_derivative(self, mu):
        """Elementwise second derivative of the link function."""
        return 6.0 / mu**4


# Dict comprehension instead of hard-coding the names again here
LINKS = {
    link.name: link
    for link in [
        Identity,
        Log,
        Logit,
        # CLogLogLink, InvSquared, Inverse, SmoothLog,
        Softplus,
    ]
}


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys", "-k link"])

    import matplotlib.pyplot as plt

    link = Softplus()

    x = np.linspace(0.1, 10)
    plt.plot(x, link(x))
    plt.plot(x, link.derivative(x))
    plt.plot(x, x)
    plt.grid(True)
    plt.show()
