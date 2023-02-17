#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:29:54 2023

@author: tommy
"""

__version__ = "0.0.0"


import logging
from generalized_additive_models.utils import set_logger

log = set_logger()
# This goes into your library somewhere
logging.getLogger("gam").addHandler(logging.NullHandler())


logging.getLogger("gam").propagate = False


from generalized_additive_models.gam import GAM
from generalized_additive_models.links import Identity
from generalized_additive_models.terms import Intercept, Linear, Spline, Tensor, TermList
from generalized_additive_models.distributions import Normal, Poisson, Binomial

__all__ = ["GAM", "Identity", "Intercept", "Linear", "Spline", "Tensor", "TermList"]
