#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:36:42 2023

@author: tommy
"""

import numpy as np
from scipy import special
from abc import ABC, abstractmethod


class Link(ABC):
    # https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
    @abstractmethod
    def link(self, mu):
        # The link function
        pass

    @abstractmethod
    def inverse_link(self, lp):
        # The inverse link function
        pass

    @abstractmethod
    def gradient(self, mu):
        # Gradient of the link function
        pass


class IdentityLink(Link):
    """g(mu) = mu"""

    name = "identity"

    def link(self, mu):
        return mu

    def inverse_link(self, lp):
        return lp

    def gradient(self, mu):
        return np.ones_like(mu)


class LogitLink(Link):
    """g(mu) = log(mu / (1 - mu))"""

    name = "logit"

    def link(self, mu, levels=1):
        if levels > 1:
            return np.log(mu) - np.log(levels - mu)
        else:
            return special.logit(mu)

    def inverse_link(self, lp, levels=1):
        # expit(x) = 1 / (1 + exp(-x))
        if levels > 1:
            return levels * special.expit(lp)
        else:
            return special.expit(lp)

    def gradient(self, mu, levels=1):
        return levels / (mu * (levels - mu))


class CLogLogLink(Link):
    """g(mu) = log(-log(1 - mu))"""

    name = "cloglog"

    def link(self, mu, levels=1):
        return np.log(np.log(levels) - np.log(levels - mu))

    def inverse_link(self, lp, levels=1):
        return levels * np.exp(-np.exp(lp)) * (np.exp(np.exp(lp)) - 1)

    def gradient(self, mu, levels=1):
        return 1 / ((levels - mu) * (np.log(levels) - np.log(levels - mu)))


class LogLink(Link):
    """g(mu) = log(mu)"""

    name = "log"

    def link(self, mu):
        return np.log(mu)

    def inverse_link(self, lp):
        return np.exp(lp)

    def gradient(self, mu):
        return 1.0 / mu


class InverseLink(Link):
    """g(mu) = 1/mu"""

    name = "inverse"

    def link(self, mu):
        return 1.0 / mu

    def inverse_link(self, lp):
        return 1.0 / lp

    def gradient(self, mu):
        return -1.0 / mu**2


class InvSquaredLink(Link):
    """g(mu) = 1/mu**2"""

    name = "inv_squared"

    def link(self, mu):
        return 1.0 / mu**2

    def inverse_link(self, lp):
        return 1.0 / np.sqrt(lp)

    def gradient(self, mu):
        return -2.0 / mu**3


# Dict comprehension instead of hard-coding the names again here
LINKS = {l.name: l for l in [IdentityLink, LogLink, LogitLink, InverseLink, InvSquaredLink, CLogLogLink]}


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "-v", "--capture=sys", "-k TestLink"])
