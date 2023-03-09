#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:13:31 2023

@author: tommy
"""

import numbers

import numpy as np
from sklearn.utils import check_scalar


def second_order_finite_difference(n, periodic=False):
    """Create a second-order finite difference matrix.

    Parameters
    ----------
    n : int
        Number of coefficients.
    periodic : bool, optional
        Whether the penalty is periodic (wraps around). The default is False.

    Returns
    -------
    np.ndarray
        A finite difference matrix.

    Examples
    --------
    >>> second_order_finite_difference(1, periodic=True)
    array([[0.]])
    >>> second_order_finite_difference(2, periodic=True)
    array([[0., 0.],
           [0., 0.]])
    >>> second_order_finite_difference(3, periodic=True)
    array([[-2.,  1.,  1.],
           [ 1., -2.,  1.],
           [ 1.,  1., -2.]])
    >>> second_order_finite_difference(3, periodic=False)
    array([[ 0.,  0.,  0.],
           [ 1., -2.,  1.],
           [ 0.,  0.,  0.]])
    >>> second_order_finite_difference(6, periodic=False)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 1., -2.,  1.,  0.,  0.,  0.],
           [ 0.,  1., -2.,  1.,  0.,  0.],
           [ 0.,  0.,  1., -2.,  1.,  0.],
           [ 0.,  0.,  0.,  1., -2.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]])
    """
    n = check_scalar(n, name="n", target_type=numbers.Integral, min_val=1, include_boundaries="left")

    if n in (1, 2):
        return np.zeros(shape=(n, n), dtype=float)

    # Set up tridiagonal
    D = (np.eye(n, k=1) + np.eye(n, k=-1) - 2 * np.eye(n)).astype(float)
    if periodic:
        # Wrap around
        D[0, -1] = 1
        D[-1, 0] = 1
        return D
    else:
        # Remove on first and last element
        D[0, :2] = [0, 0]
        D[-1, -2:] = [0, 0]
        return D


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--capture=sys", "--doctest-modules", "--maxfail=1"])
