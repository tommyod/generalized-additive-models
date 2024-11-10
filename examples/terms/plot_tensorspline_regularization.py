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
X = rng.uniform(0, 1, size=(999, 2))
y = f(X) + rng.normal(scale=0.1, size=X.shape[0])

# Plot the data set
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Dataset")
ax1.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.8)

ns = 10  # The tensor spline will fit 10 x 10 = 100 coefficients

# Create a grid to evaluate the GAM on
num_gridpoints = 50
grid = np.linspace(0, 1, num=num_gridpoints)
x1, x2 = grid, grid
X_eval = cartesian([x1, x2])
for ax_num, penalty in enumerate([1e-3, 1e-1, 1e1], 2):
    # Create and fit a GAM
    terms = Tensor(
        [
            Spline(0, num_splines=ns, penalty=penalty),
            Spline(1, num_splines=ns, penalty=penalty),
        ]
    )
    gam = GAM(terms=terms)
    gam.fit(X, y)

    # Predict and reshape outputs
    y_eval = gam.predict(X_eval)
    y_plt = y_eval.reshape(num_gridpoints, -1).T

    # Create a meshgrid for plotting, then plot
    X1_plt, X2_plt = np.meshgrid(x1, x2)
    ax2 = fig.add_subplot(2, 2, ax_num)
    ax2.set_title(f"Tensor spline (penalty={penalty})")
    CS = ax2.contour(X1_plt, X2_plt, y_plt, levels=15)
    # ax2.clabel(CS, inline=True, fontsize=8) # <- Show labels

plt.show()


# %%
# Regularization in each dimension
# ---------------------
#
# Smoothness constraints can be enforced along each dimension individually.


def f(X):
    x_1, x_2 = X[:, 0], X[:, 1]
    distance = np.sqrt((x_1 - 0.5) ** 2 + (x_2 - 0.5) ** 2)
    return np.exp(-distance)


fig = plt.figure(figsize=(6, 6), layout="constrained")

# Create data set
y = f(X) + rng.normal(scale=0.1, size=X.shape[0])

# Plot the data set
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Dataset")
ax1.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.8)


for ax_num, penalty in enumerate([1e-1, 1e0, 1e1], 2):
    # Create and fit a GAM
    terms = Tensor(
        [
            Spline(0, num_splines=ns, penalty=penalty * 1e2),
            Spline(1, num_splines=ns, penalty=penalty),
        ]
    )
    gam = GAM(terms=terms)
    gam.fit(X, y)

    # Predict and reshape outputs
    y_eval = gam.predict(X_eval)
    y_plt = y_eval.reshape(num_gridpoints, -1).T

    # Create a meshgrid for plotting, then plot
    X1_plt, X2_plt = np.meshgrid(x1, x2)
    ax2 = fig.add_subplot(2, 2, ax_num)
    ax2.set_title(f"Tensor spline (penalty={penalty})")
    CS = ax2.contour(X1_plt, X2_plt, y_plt, levels=15)

plt.show()
