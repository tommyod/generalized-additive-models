#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:36:42 2023

@author: tommy
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy import special

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


class Identity(Link):
    """g(mu) = mu"""

    name = "identity"
    domain = (-np.inf, np.inf)

    def link(self, mu):
        return mu

    def inverse_link(self, linear_prediction):
        return linear_prediction

    def derivative(self, mu):
        return np.ones_like(mu, dtype=float)

    def second_derivative(self, mu):
        return np.zeros_like(mu, dtype=float)


class Logit(Link):
    """g(mu) = log(mu / (1 - mu))"""

    name = "logit"
    domain = (0, 1)

    def __init__(self, low=0, high=1):
        assert type(low) == type(high)
        self.low = low
        self.high = high
        self.domain = (self.low, self.high)

    def link(self, mu):
        # x = log((-L + y)/(H - y))

        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        return np.log(mu - low) - np.log(high - mu)

    def inverse_link(self, linear_prediction):
        # expit(x) = 1 / (1 + exp(-x))

        low = self._validate_threshold(threshold=self.low, argument=linear_prediction)
        high = self._validate_threshold(threshold=self.high, argument=linear_prediction)

        return low + (high - low) * special.expit(linear_prediction)

    def derivative(self, mu):
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        threshold = EPSILON
        mu = np.maximum(np.minimum(mu, high - threshold), low + threshold)

        return np.exp(np.log(high - low) - np.log(high - mu) - np.log(mu - low))
        return (high - low) / ((high - mu) * (mu - low))

    def second_derivative(self, mu):
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        numerator = (low - high) * (high + low - 2 * mu)
        denominator = (high - mu) ** 2 * (low - mu) ** 2
        return numerator / denominator


class Log(Link):
    """g(mu) = log(mu)"""

    name = "log"
    domain = (0, np.inf)

    def link(self, mu):
        return np.log(mu)

    def inverse_link(self, linear_prediction):
        return np.exp(linear_prediction)

    def derivative(self, mu):
        return 1.0 / mu

    def second_derivative(self, mu):
        return -1.0 / mu**2


class SmoothLog(Link):
    """g(x) = (x^(a- 1) - 1) / (a - 1)

    where
    a -> 1       => logarithm
    a = 2        => linear function x - 1
    a > 1        => lower-log
    a < 1        => upper log

    """

    name = "smoothlog"
    domain = (0, np.inf)

    def __init__(self, a=1.5):
        self.a = a
        assert self.a <= 2
        assert self.a != 1

    def link(self, mu):
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return (mu ** (a - 1) - 1) / (a - 1)

    def inverse_link(self, linear_prediction):
        # (a*y - y + 1)**(1/(a - 1))

        a = self._validate_threshold(threshold=self.a, argument=linear_prediction)

        base = linear_prediction * (a - 1) + 1
        exponent = 1 / (a - 1)
        return base**exponent

    def derivative(self, mu):
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return mu ** (a - 2)

    def second_derivative(self, mu):
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return mu ** (a - 3) * (a - 2)


class Softplus(Link):
    """g(mu) = log(mu)"""

    # Using the Softplus Function to Construct Alternative Link Functions
    # in Generalized Linear Models and Beyond
    # https://arxiv.org/pdf/2111.14207.pdf

    name = "softplus"
    domain = (0, np.inf)

    def __init__(self, a=1):
        self.a = a
        assert a > 0

    def link(self, mu):
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return np.maximum(0, mu) + np.log1p(-np.exp(-a * np.abs(mu))) / a
        return np.log(np.exp(a * mu) - 1) / a

    def inverse_link(self, linear_prediction):
        # the inverse is the softplus

        a = self._validate_threshold(threshold=self.a, argument=linear_prediction)

        # return np.log(1 + np.exp(a * linear_prediction))/a
        return np.maximum(0, linear_prediction) + np.log1p(np.exp(-a * np.abs(linear_prediction))) / a

    def derivative(self, mu):
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return 1 / (1 - np.exp(-a * mu))

    def second_derivative(self, mu):
        a = self._validate_threshold(threshold=self.a, argument=mu)

        return -a / (4 * np.sinh(a * mu / 2) ** 2)


class CLogLogLink(Link):
    """g(mu) = log(-log(1 - mu))"""

    name = "cloglog"
    domain = (0, 1)

    def __init__(self, low=0, high=1):
        assert type(low) == type(high)
        self.low = low
        self.high = high

    def link(self, mu, levels=1):
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        return np.log(np.log(high - low) - np.log(high - mu))

    def inverse_link(self, linear_prediction, levels=1):
        low = self._validate_threshold(threshold=self.low, argument=linear_prediction)
        high = self._validate_threshold(threshold=self.high, argument=linear_prediction)

        exponential = np.exp(-np.exp(linear_prediction))

        return high - high * exponential + low * exponential

    def derivative(self, mu, levels=1):
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        return 1 / ((high - mu) * (np.log(high - low) - np.log(high - mu)))

    def second_derivative(self, mu):
        low = self._validate_threshold(threshold=self.low, argument=mu)
        high = self._validate_threshold(threshold=self.high, argument=mu)

        logterm = np.log(high - low) - np.log(high - mu)

        return (logterm - 1) / ((high - mu) ** 2 * logterm**2)


class Inverse(Link):
    """g(mu) = 1/mu"""

    name = "inverse"
    domain = (0, np.inf)

    def link(self, mu):
        return 1.0 / mu

    def inverse_link(self, linear_prediction):
        return 1.0 / linear_prediction

    def derivative(self, mu):
        return -1.0 / mu**2

    def second_derivative(self, mu):
        return 2.0 / mu**3


class InvSquared(Link):
    """g(mu) = 1/mu**2"""

    name = "inv_squared"
    domain = (0, np.inf)

    def link(self, mu):
        return 1.0 / mu**2

    def inverse_link(self, linear_prediction):
        return 1.0 / np.sqrt(linear_prediction)

    def derivative(self, mu):
        return -2.0 / mu**3

    def second_derivative(self, mu):
        return 6.0 / mu**4


# Dict comprehension instead of hard-coding the names again here
LINKS = {l.name: l for l in [Identity, Log, Logit, CLogLogLink, InvSquared, Inverse, SmoothLog, Softplus]}


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "-v", "--capture=sys", "-k link"])
