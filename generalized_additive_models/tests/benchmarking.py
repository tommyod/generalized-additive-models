#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:27:06 2023

@author: tommy
"""
from time import perf_counter

import numpy as np

from generalized_additive_models.gam import GAM
from generalized_additive_models.terms import Spline
from generalized_additive_models.links import Softplus, Logit


def create_poisson_problem(num_samples=1_000, num_features=10):
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(num_samples, num_features))

    columns = [X_col for X_col in X.T]
    powers = rng.uniform(0.8, 1.2, size=num_features)

    # Create mu as (|X_1|)**p_1 + (|X_2|)**p_2 + ...
    mu = (
        np.array([np.sign(X_j) * np.power(np.abs(X_j), p) for (X_j, p) in zip(columns, powers)]).sum(axis=0)
        / np.sqrt(num_features)
        - 1
    )
    assert np.std(mu) < 2
    assert np.std(mu) > -1
    mu = np.exp(mu)
    y = rng.poisson(lam=mu)
    y = Logit().inverse_link(rng.normal(loc=mu, scale=5))

    import matplotlib.pyplot as plt

    # plt.hist(y, bins=100)
    # plt.show()

    return X, y


def report_time(solver="pirls", num_samples=1_000, num_features=10):
    X, y = create_poisson_problem(num_samples=num_samples, num_features=num_features)

    start_time = perf_counter()
    # Create a GAM
    terms = sum(Spline(i) for i in range(num_features))
    gam = GAM(terms, link="softplus", distribution="normal", solver=solver, verbose=False, max_iter=99).fit(X, y)
    elapsed_time = perf_counter() - start_time

    print(f"Solved with solver={solver} of " + f"shape ({num_samples}, {num_features}) in {elapsed_time}")
    print(f"Objective func value: {gam.results_.iters_loss[-1]}")
    print()


if __name__ == "__main__":
    # https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html
    # $ pip install line_profiler
    # %load_ext line_profiler
    # %lprun -f GAM.fit report_time(solver="pirls", num_samples=10**4, num_features=10)

    """
    Solved with solver=pirls of shape (1000, 10) in 0.3671287909965031
    Solved with solver=lbfgsb of shape (1000, 10) in 0.38573352902312763

    Solved with solver=pirls of shape (10000, 10) in 0.8323162900051102
    Solved with solver=lbfgsb of shape (10000, 10) in 1.4314227839931846

    Solved with solver=pirls of shape (100000, 10) in 5.422825992980506
    Solved with solver=lbfgsb of shape (100000, 10) in 14.729301321000094
    """

    for num_samples in [10**3, 10**4]:
        for num_features in [10, 100]:
            for solver in GAM._parameter_constraints["solver"][0].options:
                report_time(solver=solver, num_samples=num_samples, num_features=num_features)

            print()
