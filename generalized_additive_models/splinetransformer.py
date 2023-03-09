#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 07:15:31 2023

@author: tommy
"""

import numpy as np
from sklearn.preprocessing import SplineTransformer as SklearnSplineTransformer
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


# The sklearn SplineTransformer does not extrapolate properly when the
# splines are antidifferentiated. Splines are local by nature, and sklearn
# is smart enough not to extrapolate splines 'in the middle', since they
# are zero anyway. But this is not true when the spline basis is
# antidifferentiated. In that case they go from zero to constant, and should
# be extrapolated as constants. Below we subclass the SplineTransformer
# from sklearn and fix this issue in the transform() method.
class SplineTransformer(SklearnSplineTransformer):
    def transform(self, X):
        """Transform each feature data to B-splines.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        XBS : ndarray of shape (n_samples, n_features * n_splines)
            The matrix of features, where n_splines is the number of bases
            elements of the B-splines, n_knots + degree - 1.
        """
        check_is_fitted(self)

        X = self._validate_data(X, reset=False, accept_sparse=False, ensure_2d=True)

        n_samples, n_features = X.shape
        n_splines = self.bsplines_[0].c.shape[1]
        degree = self.degree

        # Note that scipy BSpline returns float64 arrays and converts input
        # x=X[:, i] to c-contiguous float64.
        n_out = self.n_features_out_ + n_features * (1 - self.include_bias)
        if X.dtype in FLOAT_DTYPES:
            dtype = X.dtype
        else:
            dtype = np.float64
        XBS = np.zeros((n_samples, n_out), dtype=dtype, order=self.order)

        for i in range(n_features):
            spl = self.bsplines_[i]

            if self.extrapolation in ("continue", "error", "periodic"):
                if self.extrapolation == "periodic":
                    # With periodic extrapolation we map x to the segment
                    # [spl.t[k], spl.t[n]].
                    # This is equivalent to BSpline(.., extrapolate="periodic")
                    # for scipy>=1.0.0.
                    n = spl.t.size - spl.k - 1
                    # Assign to new array to avoid inplace operation
                    x = spl.t[spl.k] + (X[:, i] - spl.t[spl.k]) % (spl.t[n] - spl.t[spl.k])
                else:
                    x = X[:, i]

                XBS[:, (i * n_splines) : ((i + 1) * n_splines)] = spl(x)

            else:
                xmin = spl.t[degree]
                xmax = spl.t[-degree - 1]
                mask = (xmin <= X[:, i]) & (X[:, i] <= xmax)
                XBS[mask, (i * n_splines) : ((i + 1) * n_splines)] = spl(X[mask, i])

            # Note for extrapolation:
            # 'continue' is already returned as is by scipy BSplines
            if self.extrapolation == "error":
                # BSpline with extrapolate=False does not raise an error, but
                # output np.nan.
                if np.any(np.isnan(XBS[:, (i * n_splines) : ((i + 1) * n_splines)])):
                    raise ValueError("X contains values beyond the limits of the knots.")
            elif self.extrapolation == "constant":
                # Set all values beyond xmin and xmax to the value of the
                # spline basis functions at those two positions.
                # Only the first degree and last degree number of splines
                # have non-zero values at the boundaries.

                # spline values at boundaries
                f_min = spl(xmin)
                f_max = spl(xmax)
                mask = X[:, i] < xmin
                if np.any(mask):
                    XBS[mask, (i * n_splines) : (i * n_splines + len(f_min))] = f_min[:]

                mask = X[:, i] > xmax
                if np.any(mask):
                    XBS[
                        mask,
                        ((i + 1) * n_splines - len(f_min)) : ((i + 1) * n_splines),
                    ] = f_max[:]

            elif self.extrapolation == "linear":
                # Continue the degree first and degree last spline bases
                # linearly beyond the boundaries, with slope = derivative at
                # the boundary.
                # Note that all others have derivative = value = 0 at the
                # boundaries.

                # spline values at boundaries
                f_min, f_max = spl(xmin), spl(xmax)
                # spline derivatives = slopes at boundaries
                fp_min, fp_max = spl(xmin, nu=1), spl(xmax, nu=1)
                # Compute the linear continuation.
                if degree <= 1:
                    # For degree=1, the derivative of 2nd spline is not zero at
                    # boundary. For degree=0 it is the same as 'constant'.
                    degree += 1
                for j in range(len(f_min)):
                    mask = X[:, i] < xmin
                    if np.any(mask):
                        XBS[mask, i * n_splines + j] = f_min[j] + (X[mask, i] - xmin) * fp_min[j]

                    mask = X[:, i] > xmax
                    if np.any(mask):
                        k = n_splines - 1 - j
                        XBS[mask, i * n_splines + k] = f_max[k] + (X[mask, i] - xmax) * fp_max[k]

        if self.include_bias:
            return XBS
        else:
            # We throw away one spline basis per feature.
            # We chose the last one.
            indices = [j for j in range(XBS.shape[1]) if (j + 1) % n_splines != 0]
            return XBS[:, indices]


if __name__ == "__main__":
    # from sklearn.preprocessing import SplineTransformer
    import matplotlib.pyplot as plt

    X = np.linspace(-1, 2).reshape(-1, 1)

    t = SplineTransformer(n_knots=3, extrapolation="constant", degree=3)
    t.fit(X)
    t.bsplines_[0] = t.bsplines_[0].antiderivative(2)
    t.degree += 2

    # increase = np.max()
    derivative = np.max(t.bsplines_[0](1, nu=1))
    stabilizer = X * derivative

    plt.plot(X, t.transform(X) - 0, alpha=0.5, lw=4)
    plt.show()

    X = np.linspace(-1.5, 2.5, num=2**7).reshape(-1, 1)
    X = np.sort(X, axis=0)

    plt.plot(X, t.transform(X), alpha=0.5, lw=4)
    plt.axvline(x=2, ls="--", alpha=0.5, color="black")
    plt.axvline(x=-1, ls="--", alpha=0.5, color="black")
