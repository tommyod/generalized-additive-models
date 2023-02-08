#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 07:56:33 2023

@author: tommy
"""

import scipy as sp
import numpy as np


def tensor_product(a, b, reshape=True):
    """
    compute the tensor protuct of two matrices a and b

    if a is (n, m_a), b is (n, m_b),
    then the result is
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Parameters
    ---------
    a : array-like of shape (n, m_a)
    b : array-like of shape (n, m_b)

    reshape : bool, default True
        whether to reshape the result to be 2-dimensional ie
        (n, m_a * m_b)
        or return a 3-dimensional tensor ie
        (n, m_a, m_b)

    Returns
    -------
    dense np.ndarray of shape
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Examples
    --------
    >>> A = np.eye(3, dtype=int)
    >>> B = np.arange(9).reshape(3, 3)
    >>> tensor_product(A, B)
    array([[0, 1, 2, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 3, 4, 5, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 6, 7, 8]])
    >>> A = np.diag([1, 2, 3])
    >>> A[0, :] = [1, 2, 3]
    >>> tensor_product(A, B, reshape=True)
    array([[ 0,  1,  2,  0,  2,  4,  0,  3,  6],
           [ 0,  0,  0,  6,  8, 10,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 18, 21, 24]])
    >>> tensor_product(A, B, reshape=False)
    array([[[ 0,  1,  2],
            [ 0,  2,  4],
            [ 0,  3,  6]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 6,  8, 10],
            [ 0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  0],
            [18, 21, 24]]])


    """
    assert a.ndim == 2, f"matrix a must be 2-dimensional, but found {a.ndim} dimensions"
    assert b.ndim == 2, f"matrix b must be 2-dimensional, but found {b.nim} dimensions"

    na, ma = a.shape
    nb, mb = b.shape

    if na != nb:
        raise ValueError("both arguments must have the same number of samples")

    if sp.sparse.issparse(a):
        a = a.A

    if sp.sparse.issparse(b):
        b = b.A

    tensor = a[..., :, None] * b[..., None, :]

    if reshape:
        return tensor.reshape(na, ma * mb)

    return tensor


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--capture=sys", "--doctest-modules", "--maxfail=1"])
