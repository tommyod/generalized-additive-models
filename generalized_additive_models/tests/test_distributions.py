#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:01:43 2023

@author: tommy
"""

import numpy as np
import pytest
from sklearn.base import clone

from generalized_additive_models.distributions import DISTRIBUTIONS, Bernoulli, Binomial, Exponential, Gamma, Normal


class TestSklearnCompatibility:
    def test_cloning_with_sklearn_clone(self):
        normal = Normal(10)
        cloned = clone(normal)
        normal.scale = 1

        assert cloned.scale == 10

    def test_that_get_and_set_params_works(self):
        normal = Normal(13)
        assert normal.get_params(True) == {"scale": 13}
        assert normal.get_params(False) == {"scale": 13}

        normal.set_params(scale=1)
        assert normal.get_params() == {"scale": 1}


class TestDistributionProperties:
    @pytest.mark.parametrize("distr_class", list(DISTRIBUTIONS.values()))
    @pytest.mark.parametrize("mu", [0.2, 0.5, 0.95])  # Common range for all distributions
    @pytest.mark.parametrize("scale", [0.5, 1, 2])
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_that_theoretical_mean_and_variance_matches_samples(self, distr_class, mu, scale, seed):
        distribution = distr_class(scale=scale)
        samples = distribution.sample(mu=mu, size=100_000, random_state=seed)

        assert np.isclose(mu, samples.mean(), rtol=0.05)
        assert np.isclose(distribution.variance(mu), samples.var(), rtol=0.04)

    @pytest.mark.parametrize("distr_class", list(DISTRIBUTIONS.values()))
    @pytest.mark.parametrize("mu", [0.2, 0.5, 0.95])  # Common range for all distributions
    @pytest.mark.parametrize("scale", [0.5, 1, 2])
    def test_that_theoretical_mean_matches_scipy_mean(self, distr_class, mu, scale):
        """This property should hold for all values of the scale."""

        distribution = distr_class(scale=scale)
        assert np.isclose(distribution.to_scipy(mu).mean(), mu)

    @pytest.mark.parametrize("distr_class", list(DISTRIBUTIONS.values()))
    @pytest.mark.parametrize("mu", [0.2, 0.5, 0.95])  # Common range for all distributions
    @pytest.mark.parametrize("scale", [0.5, 1, 2])
    def test_that_theoretical_variance_matches_scipy_variance(self, distr_class, mu, scale):
        """This property should hold for all values of the scale."""

        distribution = distr_class(scale=scale)
        assert np.isclose(distribution.to_scipy(mu).var(), distribution.variance(mu))

    @pytest.mark.parametrize("distr_class", list(DISTRIBUTIONS.values()))
    @pytest.mark.parametrize("scale", [0.5, 1, 2])
    def test_that_scipy_deviance_matches_implemented_deviance(self, distr_class, scale):
        """This property should hold for all values of the scale."""

        distribution = distr_class(scale=scale)
        mu = np.array([0.083, 0.083, 0.192, 0.295, 0.34, 0.498, 0.987, 0.99994])

        # Binomial with 1 trial is Bernoulli, which has support {0, 1}
        # However, gamma is only defined for (0, \inf)
        # So I have to choose a vector of ones here for all tests to pass
        y = np.ones_like(mu)

        deviance = distribution.deviance(mu=mu, y=y)

        # Saturated model and actual model
        if hasattr(distribution.to_scipy(mu=y), "logpdf"):
            pdf_sat = distribution.to_scipy(mu=y).logpdf(y)
            pdf_obs = distribution.to_scipy(mu=mu).logpdf(y)
        else:
            pdf_sat = distribution.to_scipy(mu=y).logpmf(y)
            pdf_obs = distribution.to_scipy(mu=mu).logpmf(y)

        # This is the definition of deviance
        # https://en.wikipedia.org/wiki/Deviance_(statistics)
        assert np.allclose(deviance, 2 * (pdf_sat - pdf_obs))

    def test_distributions_that_are_special_cases_of_others(self):
        mu = np.array([0.083, 0.083, 0.192, 0.295, 0.34, 0.498, 0.987, 0.99994])
        y = np.ones_like(mu)

        # Exponential is a special case of Gamma
        exponential = Exponential()
        gamma_as_exponential = Gamma(scale=1)

        logpdf1 = exponential.to_scipy(mu=y).logpdf(y)
        logpdf2 = gamma_as_exponential.to_scipy(mu=y).logpdf(y)
        assert np.allclose(logpdf1, logpdf2)

        # Bernoulli is a special case of Binomial
        bernoulli = Bernoulli()
        binomial_as_bernoulli = Binomial(trials=1)

        logpdf1 = bernoulli.to_scipy(mu=y).logpmf(y)
        logpdf2 = binomial_as_bernoulli.to_scipy(mu=y).logpmf(y)
        assert np.allclose(logpdf1, logpdf2)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            "--capture=sys",
            "--doctest-modules",
            "--maxfail=1",
        ]
    )
