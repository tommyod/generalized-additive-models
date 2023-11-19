#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:33:49 2023

@author: tommy
"""

from abc import ABC, abstractmethod
from numbers import Real

import numpy as np
import scipy as sp
from scipy.special import rel_entr
from sklearn.base import BaseEstimator
from sklearn.utils import check_consistent_length
from sklearn.utils._param_validation import Interval

MACHINE_EPSILON = np.finfo(float).eps
EPSILON = np.sqrt(MACHINE_EPSILON)


class Distribution(ABC):
    @abstractmethod
    def sample(self, mu):
        pass

    def variance(self, mu):
        return self.V(mu) * self.scale

    @abstractmethod
    def V(self, mu):
        pass

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.get_params() == other.get_params()


class Normal(Distribution, BaseEstimator):
    name = "normal"
    domain = (-np.inf, np.inf)
    continuous = True

    _parameter_constraints: dict = {
        "scale": [Interval(Real, 0.0, None, closed="neither"), None],
    }

    def __init__(self, scale=None):
        """Create a Normal distribution.


        Parameters
        ----------
        scale : float or None, optional
            If None, will be set by the GAM. The default is None.

        Returns
        -------
        None.

        """
        self.scale = scale

    def log_pdf(self, y, mu):
        standard_deviation = np.sqrt(self.variance(mu))
        return sp.stats.norm.logpdf(y, loc=mu, scale=standard_deviation)

    def V(self, mu):
        return np.ones_like(mu, dtype=float)

    def V_derivative(self, mu):
        return np.zeros_like(mu, dtype=float)

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        check_consistent_length(y, mu, sample_weight)

        deviance = (y - mu) ** 2
        if scaled and self.scale is not None:
            deviance = deviance / self.scale

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        return deviance * sample_weight

    def sample(self, mu, size=None):
        standard_deviation = np.sqrt(self.variance(mu))
        return np.random.normal(loc=mu, scale=standard_deviation, size=size)

    def to_scipy(self, mu):
        standard_deviation = np.sqrt(self.variance(mu))
        return sp.stats.norm(loc=mu, scale=standard_deviation)


class Poisson(Distribution, BaseEstimator):
    """
    Poisson Distribution
    """

    name = "poisson"
    domain = (0, np.inf)
    continuous = True
    scale = 1

    def __init__(self):
        pass

    def log_pdf(self, y, mu):
        return sp.stats.poisson.logpmf(y, mu=mu)

    def V(self, mu):
        return mu

    def V_derivative(self, mu):
        return np.ones_like(mu, dtype=float)

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        check_consistent_length(y, mu, sample_weight)

        # rel_entr(y, mu) := y log(y / mu)
        deviance = 2 * (rel_entr(y, mu) - (y - mu))
        if scaled and self.scale:
            deviance = deviance / self.scale

        if sample_weight is None:
            return deviance

        return deviance * sample_weight

    def sample(self, mu, size=None):
        return np.random.poisson(lam=mu, size=size)

    def to_scipy(self, mu):
        return sp.stats.poisson(mu=mu)


class Binomial(Distribution, BaseEstimator):
    """
    Binomial Distribution
    """

    name = "binomial"
    scale = 1

    def __init__(self, trials=1):
        """
        creates an instance of the Binomial class

        Parameters
        ----------
        trials : int of None, default: 1
            number of trials in the binomial distribution

        Returns
        -------
        self
        """
        assert isinstance(trials, (int, np.ndarray)), "trials must be an integer"
        if isinstance(trials, np.ndarray):
            assert np.all(trials >= 1), "trials must be >= 1"
        else:
            assert trials >= 1, "trials must be >= 1"
        self.trials = trials

    @property
    def domain(self):
        domain = (0, self.trials)
        return domain

    def log_pdf(self, y, mu):
        n = self.trials
        p = mu / self.trials
        return sp.stats.binom.logpmf(y, n, p)

    def variance(self, mu):
        return self.V(mu) * self.scale

    def V(self, mu):
        threshold = EPSILON
        mu = np.maximum(np.minimum(mu, self.trials - threshold), 0 + threshold)

        return mu * (1 - mu / self.trials)

    def V_derivative(self, mu):
        return 1 - 2 * (mu / self.trials)

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        check_consistent_length(y, mu, sample_weight)

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        deviance = 2 * (rel_entr(y, mu) + rel_entr(self.trials - y, self.trials - mu))

        if scaled and self.scale:
            deviance = deviance / self.scale

        return deviance * sample_weight

    def sample(self, mu, size=None):
        n = self.trials
        p = mu / self.trials
        return np.random.binomial(n=n, p=p, size=size)

    def to_scipy(self, mu):
        n = self.trials
        p = mu / self.trials
        return sp.stats.binom(n, p)


class Gamma(Distribution, BaseEstimator):
    """
    Gamma Distribution
    """

    name = "gamma"
    domain = (0, np.inf)

    def __init__(self, scale=None):
        self.scale = scale

    def log_pdf(self, y, mu):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
        # The parametrization in scipy, vs. Wood table 3.1 in page 104 is
        # x = y
        # a = nu
        # scale = mu / nu = mu * scale
        nu = 1 / self.scale
        return sp.stats.gamma.logpdf(x=y, a=nu, scale=mu / nu)

    def variance(self, mu):
        return self.V(mu) * self.scale

    def V(self, mu):
        return mu**2

    def V_derivative(self, mu):
        return 2 * mu

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        check_consistent_length(y, mu, sample_weight)

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        deviance = 2 * ((y - mu) / mu - np.log(y / mu))

        if scaled and self.scale:
            deviance = deviance / self.scale

        return deviance * sample_weight

    def sample(self, mu, size=None):
        nu = 1 / self.scale
        return sp.stats.gamma(a=nu, scale=mu / nu).rvs(size=size)


class InvGauss(Distribution, BaseEstimator):
    """
    Inverse Gaussian Distribution
    """

    name = "inv_gauss"
    domain = (0, np.inf)

    def __init__(self, scale=None):
        self.scale = scale

    def log_pdf(self, y, mu):
        gamma = 1 / self.scale
        return sp.stats.invgauss.logpdf(y, mu, scale=gamma**1 / 3)

    def V(self, mu):
        return mu**3

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        check_consistent_length(y, mu, sample_weight)

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        deviance = ((y - mu) ** 2) / (mu**2 * y)

        if scaled and self.scale:
            deviance = deviance / self.scale

        return deviance

    def sample(self, mu, size=None):
        gamma = 1 / self.scale
        return sp.stats.invgauss(mu, scale=gamma**1 / 3).rvs(None)


DISTRIBUTIONS = {
    dist.name: dist
    for dist in [
        Normal,
        Poisson,
        Binomial,
        Gamma,
        #  InvGaussDist,
    ]
}


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "-v", "--capture=sys", "-k TestLink"])
