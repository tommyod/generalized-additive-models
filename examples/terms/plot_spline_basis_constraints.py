"""
=============================
Spline basis with constraints
=============================

Plot a spline basis with constraints.

"""

import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models import Spline

# Create data
rng = np.random.default_rng(42)
x = np.linspace(-3, 3, num=2**10)
X = x[:, np.newaxis]

constraints = [
    "increasing",
    "decreasing",
    "convex",
    "concave",
    "increasing-convex",
    "increasing-concave",
    "decreasing-convex",
    "decreasing-concave",
]

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
fig.suptitle("Spline bases for constrained splines", y=0.95, fontsize=14)

for constraint, ax in zip(constraints, axes.ravel()):
    ax.set_title(constraint)

    # Create a Spline
    spline = Spline(0, num_splines=6, degree=3, constraint=constraint)
    X_transformed = spline.fit_transform(X)

    # Plot the Spline basis
    ax.plot(x, X_transformed)
    ax.set_yticks([])

fig.tight_layout()
plt.show()

# %%
# Draw random functions
# ---------------------
#
# As long as the coefficients are positive, the constraint will be obeyed.

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
fig.suptitle("Sampled functions for constrained splines", y=0.95, fontsize=14)
num_splines = 6
num_functions = 10

for constraint, ax in zip(constraints, axes.ravel()):
    ax.set_title(constraint)
    ax.set_yticks([])

    # Create a Spline
    spline = Spline(0, num_splines=6, degree=3, constraint=constraint)
    X_transformed = spline.fit_transform(X)

    # Sample random positive coefficients and plot functions
    for coef in rng.exponential(size=(num_functions, num_splines)):
        ax.plot(x, X_transformed @ coef, color="black", alpha=0.5)

fig.tight_layout()
plt.show()
