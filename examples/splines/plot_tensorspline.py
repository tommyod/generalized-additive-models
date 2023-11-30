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
    # The smooth XOR function, but squaring one variable
    x_1 = X[:, 0] ** 2
    x_2 = X[:, 1]
    return (x_1 + x_2) * (1 - (x_1 * x_2))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# Create data set
rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(3333, 2))
y = f(X)

# Plot the data set
ax1.set_title("Dataset")
ax1.scatter(X[:, 0], X[:, 1], c=y, s=10)

# Fit a GAM
num_splines = 10  # The tensor spline will fit 10 x 10 = 100 coefficients
terms = Tensor(Spline(0, num_splines=num_splines) + Spline(1, num_splines=num_splines))
gam = GAM(terms=terms)
gam.fit(X, y)

# Create a grid to evaluate the GAM on
num_gridpoints = 50
x1, x2 = np.linspace(0, 1, num=num_gridpoints), np.linspace(0, 1, num=num_gridpoints)
X_eval = cartesian([x1, x2])
y_eval = gam.predict(X_eval)

# Create a meshgrid for plotting
X_plt, Y_plt = np.meshgrid(x1, x2)
ax2.set_title("Tensor spline")
CS = ax2.contour(X_plt, Y_plt, y_eval.reshape(num_gridpoints, -1).T, levels=15)
ax2.clabel(CS, inline=True, fontsize=10)

fig.tight_layout()
plt.show()
