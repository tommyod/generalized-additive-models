#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:05:41 2023

@author: tommy
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from generalized_additive_models.terms import Spline, Linear


X = np.sort(np.abs(np.random.randn(999, 1)), axis=0)
y = np.sin(X).ravel() + np.random.randn(999) / 10


plt.scatter(X, y)

basis = Spline(0, num_splines=10, knots="uniform", edges=(0, 5), extrapolation="constant").transform(X)

plt.plot(X, basis)


class GAM(BaseEstimator):
    def __init__(self, terms=None, tol=1e-5):
        self.terms = terms
        self.tol = tol

    def fit(self, X, y):
        self.model = Ridge().fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


gam = GAM(terms=Spline(0) + Spline(1))
if False:
    print(gam.get_params())

    print(gam.set_params(**{"tol": 0.1, "terms__data__": 1}))

    print(gam)

from sklearn.base import clone


clone(gam)


from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

cross_val_score(gam, X, y, verbose=10)


from sklearn.model_selection import GridSearchCV


search = GridSearchCV(
    gam,
    param_grid={"terms__0__penalty": [1, 2, 3]},
    scoring=None,
    n_jobs=1,
    refit=True,
    cv=10,
    verbose=99,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
)


search.fit(X, y)

print("===========================================")


search = GridSearchCV(
    gam,
    param_grid={"terms__0": [Spline(penalty=99), Spline(penalty=2)]},
    scoring=None,
    n_jobs=1,
    refit=True,
    cv=10,
    verbose=99,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
)


search.fit(X, y)


sorted(search.cv_results_.keys())

print(
    (Spline(0, penalty=1, by=1) + Linear(1)).set_params(
        **{
            "0__by": 1,
            "0__degree": 3,
            "0__edges": None,
            "0": Spline(by=1, feature=0),
            "1__by": None,
            "1__feature": 1,
            "1__penalty": 1,
            "1": Linear(feature=1),
        }
    )
)
