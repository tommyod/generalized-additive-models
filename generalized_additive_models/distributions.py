#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:33:49 2023

@author: tommy
"""

"""
Distributions
"""

from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
import scipy as sp
from scipy.special import rel_entr
from sklearn.base import BaseEstimator
from sklearn.utils import check_consistent_length

MACHINE_EPSILON = np.finfo(float).eps
EPSILON = np.sqrt(MACHINE_EPSILON)


def multiply_weights(deviance):
    @wraps(deviance)
    def multiplied(self, y, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return deviance(self, y, mu, **kwargs) * weights

    return multiplied


def divide_weights(V):
    @wraps(V)
    def divided(self, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return V(self, mu, **kwargs) / weights

    return divided


class Distribution(ABC):
    """
    base distribution class
    """

    def phi(self, y, mu, edof, weights=None):
        """
        GLM scale parameter.
        for Binomial and Poisson families this is unity
        for Normal family this is variance

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        edof : float
            estimated degrees of freedom
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        scale : estimated model scale
        """
        if self.scale is not None:
            return self.scale

        if weights is None:
            weights = np.ones_like(y, dtype=float)

        if not (len(y) == len(mu) == len(weights)):
            msg = f"Lengths of y, mu and weights did not match: {len(y)} {len(mu)} {len(weights)}"
            raise ValueError(msg)

        # This is the Pearson statistic at its expected value
        # See Section 3.1.5 in Wood, 2nd ed
        # The method V() is defined by subclasses
        return np.sum(weights * (y - mu) ** 2 / self.V(mu)) / (len(mu) - edof)

    @abstractmethod
    def sample(self, mu):
        pass

    @abstractmethod
    def V(self, mu):
        pass


class Normal(Distribution, BaseEstimator):
    name = "normal"
    domain = (-np.inf, np.inf)
    continuous = True

    def __init__(self, scale=None):
        """
        creates an instance of the NormalDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        self.scale = scale

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """

        if weights is None:
            weights = np.ones_like(mu, dtype=float)

        scale = self.scale / weights
        return sp.stats.norm.logpdf(y, loc=mu, scale=scale)

    def V(self, mu, sample_weight=None):
        check_consistent_length(mu, sample_weight)

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        return np.ones_like(mu) / sample_weight

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        check_consistent_length(y, mu, sample_weight)

        deviance = (y - mu) ** 2
        if scaled and self.scale:
            deviance = deviance / self.scale

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        return deviance * sample_weight

    def sample(self, mu):
        standard_deviation = self.scale or 1.0
        return np.random.normal(loc=mu, scale=standard_deviation, size=None)


class Poisson(Distribution, BaseEstimator):
    """
    Poisson Distribution
    """

    name = "poisson"
    domain = (0, np.inf)
    continuous = True

    def __init__(self, scale=1):
        self.scale = scale

    def log_pdf(self, y, mu, weights=None):
        if weights is None:
            weights = np.ones_like(mu, dtype=float)
        # in Poisson regression weights are proportional to the exposure
        # so we want to pump up all our predictions
        # NOTE: we assume the targets are counts, not rate.
        # ie if observations were scaled to account for exposure, they have
        # been rescaled before calling this function.
        # since some samples have higher exposure,
        # they also need to have higher variance,
        # we do this by multiplying mu by the weight=exposure
        mu = mu * weights
        return sp.stats.poisson.logpmf(y, mu=mu)

    def V(self, mu, weights=None):
        if weights is None:
            weights = np.ones_like(mu, dtype=float)
        return mu / weights

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        check_consistent_length(y, mu, sample_weight)

        deviance = 2 * (rel_entr(y, mu) - (y - mu))
        if scaled and self.scale:
            deviance = deviance / self.scale

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        return deviance * sample_weight

    def sample(self, mu):
        return np.random.poisson(lam=mu, size=None)


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
        assert isinstance(trials, int), "trials must be an integer"
        assert trials > 0, "trials must be >= 1"
        self.trials = trials

    @property
    def domain(self):
        domain = (0, self.trials)
        return domain

    def log_pdf(self, y, mu, *, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)

        n = self.trials
        p = mu / self.trials
        return sp.stats.binom.logpmf(y, n, p)

    def V(self, mu, weights=None):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """

        threshold = EPSILON
        mu = np.maximum(np.minimum(mu, 1 - threshold), 0 + threshold)

        V = mu * (1 - mu / self.trials)

        if weights is None:
            weights = np.ones_like(mu, dtype=float)

        return V / weights

    def deviance(self, *, y, mu, sample_weight=None, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        check_consistent_length(y, mu, sample_weight)

        if sample_weight is None:
            sample_weight = np.ones_like(mu, dtype=float)

        deviance = 2 * (rel_entr(y, mu) + rel_entr(self.trials - y, self.trials - mu))

        return deviance * sample_weight

    def sample(self, mu):
        """
        Return random samples from this Binomial distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        number_of_trials = self.trials
        success_probability = mu / number_of_trials
        return np.random.binomial(n=number_of_trials, p=success_probability, size=None)


class GammaDist(Distribution, BaseEstimator):
    """
    Gamma Distribution
    """

    name = "gamma"
    domain = (0, np.inf)

    def __init__(self, scale=None):
        """
        creates an instance of the GammaDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super().__init__(scale=scale)

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        nu = weights / self.scale
        return sp.stats.gamma.logpdf(x=y, a=nu, scale=mu / nu)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**2

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * ((y - mu) / mu - np.log(y / mu))

        if scaled and self.scale:
            dev = dev / self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Gamma distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        # in numpy.random.gamma, `shape` is the parameter sometimes denoted by
        # `k` that corresponds to `nu` in S. Wood (2006) Table 2.1
        shape = 1.0 / self.scale
        # in numpy.random.gamma, `scale` is the parameter sometimes denoted by
        # `theta` that corresponds to mu / nu in S. Wood (2006) Table 2.1
        scale = mu / shape
        return np.random.gamma(shape=shape, scale=scale, size=None)


class InvGaussDist(Distribution, BaseEstimator):
    """
    Inverse Gaussian (Wald) Distribution
    """

    name = "inv_gauss"
    domain = (0, np.inf)

    def __init__(self, scale=None):
        """
        creates an instance of the InvGaussDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super().__init__(scale=scale)

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        gamma = weights / self.scale
        return sp.stats.invgauss.logpdf(y, mu, scale=1.0 / gamma)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**3

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = ((y - mu) ** 2) / (mu**2 * y)

        if scaled and self.scale:
            dev = dev / self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Inverse Gaussian (Wald) distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        return np.random.wald(mean=mu, scale=self.scale, size=None)


DISTRIBUTIONS = {
    dist.name: dist
    for dist in [
        Normal,
        Poisson,
        Binomial,
        #  GammaDist,
        #  InvGaussDist,
    ]
}


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "-v", "--capture=sys", "-k TestLink"])
