#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generalized Additive Models
---------------------------

A generalized additive model (GAM) is a generalized linear model in which the 
linear target variable depends linearly on unknown smooth functions (splines)
of some features, and interest focuses on inference about these smooth functions.

The model relates a target variable y, to some predictor variables, X. 
An exponential family distribution is specified for y (e.g. Normal, Binomial, Poisson) 
along with a link function g (e.g. Identity or Log) relating the expected value 
of y to the features via a structure such as

    g(E(y)) = f_1(x_1) + f_2(x_2) + ... + f_m(x_m)

The functions f_i are regression splines, to be estimated by non-parametric means.

Terms
-----

A model is constructed from a dataseting using Terms. These construct Spline
basis functions that the GAM optimizes.

>>> import numpy as np
>>> from generalized_additive_models import Spline
>>> X = np.linspace(0, 1, num=9).reshape(-1, 1)
>>> spline = Spline(0, num_splines=3, degree=0)
>>> spline.fit_transform(X)
array([[ 0.66666667, -0.33333333, -0.33333333],
       [ 0.66666667, -0.33333333, -0.33333333],
       [ 0.66666667, -0.33333333, -0.33333333],
       [-0.33333333,  0.66666667, -0.33333333],
       [-0.33333333,  0.66666667, -0.33333333],
       [-0.33333333,  0.66666667, -0.33333333],
       [-0.33333333, -0.33333333,  0.66666667],
       [-0.33333333, -0.33333333,  0.66666667],
       [-0.33333333, -0.33333333,  0.66666667]])



"""

__version__ = "0.0.0"
__name__ = "generalized-additive-models"


from generalized_additive_models.distributions import Binomial, Normal, Poisson
from generalized_additive_models.gam import GAM
from generalized_additive_models.links import Identity, Log, Logit
from generalized_additive_models.terms import Categorical, Intercept, Linear, Spline, Tensor, TermList

__all__ = [
    "Binomial",
    "Categorical",
    "GAM",
    "Identity",
    "Intercept",
    "Linear",
    "Log",
    "Logit",
    "Normal",
    "Poisson",
    "Spline",
    "Tensor",
    "TermList",
]
