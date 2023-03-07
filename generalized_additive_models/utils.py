#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 07:56:33 2023

@author: tommy
"""

import logging
import sys

import numpy as np
import scipy as sp


def cartesian(arrays):
    """
    Generate a cartesian product of input arrays.
    Adapted from:
        https://github.com/scikit-learn/scikit-learn/blob/
        master/sklearn/utils/extmath.py#L489

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = np.empty_like(ix, dtype=dtype)

    for n, _ in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def set_logger():
    log = logging.getLogger("gam")

    # https://docs.python.org/3/library/logging.html#logging.Logger
    log.propagate = False  # Do not propagate to top-level logger

    # Create handler
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)

    # Create and set formatter
    formatter = logging.Formatter("%(levelname)-8s: %(message)s")
    stream_handler.setFormatter(formatter)

    # Set handler
    if not log.handlers:
        log.addHandler(stream_handler)

    return log


# log = set_logger()


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
    >>> import numpy as np
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

    A tensor where one dimension is integrated:

    >>> # This is integrated
    >>> A = np.array([[1., 0., 0.],
    ...               [1., 1., 0.],
    ...               [1., 1., 1.],
    ...               [1., 0., 0.],
    ...               [1., 1., 0.],
    ...               [1., 1., 1.],
    ...               [1., 0., 0.],
    ...               [1., 1., 0.],
    ...               [1., 1., 1.]])
    >>> B = np.array([[1, 0, 0],
    ...               [1, 0, 0],
    ...               [1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [0, 0, 1],
    ...               [0, 0, 1]])
    >>> tensor_product(A, B, reshape=True)
    array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 1., 0., 0., 0., 0., 0.],
           [1., 0., 0., 1., 0., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 1., 0., 0., 1., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0., 1., 0., 0., 1.]])


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
