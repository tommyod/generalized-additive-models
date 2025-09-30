"""
====================================
Tensor spline
====================================

Plot a Tensor spline.

"""

import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models import GAM, Spline, Tensor
from generalized_additive_models.utils import cartesian


def f(X):
    """A non-additive function."""
    assert X.shape[1] == 2
    x_1 = X[:, 0] ** 2
    x_2 = X[:, 1]
    return (x_1 + x_2) * (1 - (x_1 * x_2))


fig = plt.figure(figsize=(6, 6), layout="constrained")

# Create data set
rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(3333, 2))
y = f(X) + rng.normal(scale=0.01, size=X.shape[0])

# Plot the data set
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Dataset")
ax1.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.5)

# Fit a GAM
ns = 10  # The tensor spline will fit 10 x 10 = 100 coefficients
terms = Tensor(Spline(0, num_splines=ns) + Spline(1, num_splines=ns))
gam = GAM(terms=terms)
gam.fit(X, y)

# Create a grid to evaluate the GAM on
num_gridpoints = 50
x1, x2 = np.linspace(0, 1, num=num_gridpoints), np.linspace(0, 1, num=num_gridpoints)
X_eval = cartesian([x1, x2])
y_eval = gam.predict(X_eval)
y_plt = y_eval.reshape(num_gridpoints, -1).T

# Create a meshgrid for plotting
X1_plt, X2_plt = np.meshgrid(x1, x2)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Tensor spline")
CS = ax2.contour(X1_plt, X2_plt, y_plt, levels=15)
ax2.clabel(CS, inline=True, fontsize=8)

# Plot surface
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_title("Surface plot")
ax3.plot_surface(
    X1_plt,
    X2_plt,
    y_plt,
    rstride=3,
    cstride=3,
    cmap="viridis",
    linewidth=1,
    antialiased=True,
)
ax3.zaxis.set_ticklabels([])

# Plot contour on top of data
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("Fitted Tensor on top of data")
ax4.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.2)
ax4.contour(X1_plt, X2_plt, y_plt, levels=15, colors="black")

plt.show()
