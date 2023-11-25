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






https://arxiv.org/pdf/2006.06466.pdf


Pya, N., Wood, S.N. 
Shape constrained additive models. 
Stat Comput 25, 543–559 (2015). https://doi.org/10.1007/s11222-013-9448-7
https://link.springer.com/article/10.1007/s11222-013-9448-7

"""

import importlib.metadata
import warnings

from generalized_additive_models.distributions import Binomial, Normal, Poisson
from generalized_additive_models.gam import GAM, ExpectileGAM
from generalized_additive_models.links import Identity, Log, Logit, Softplus
from generalized_additive_models.terms import Categorical, Intercept, Linear, Spline, Tensor, TermList

__name__ = "generalized-additive-models"
__version__ = importlib.metadata.version(__name__)

__all__ = [
    "Binomial",
    "Categorical",
    "GAM",
    "ExpectileGAM",
    "Identity",
    "Intercept",
    "Linear",
    "Log",
    "Logit",
    "Normal",
    "Poisson",
    "Spline",
    "Softplus",
    "Tensor",
    "TermList",
]


message = f"""\nThank you for using {__name__}, version {__version__}.
Until version 1.0.0 is released, the package and API should be considered unstable.
You are welcome to use the package. Report bugs and join the discussion on GitHub:
https://github.com/tommyod/generalized-additive-models"""


warnings.warn(message)
