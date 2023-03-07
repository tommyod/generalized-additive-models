# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Constraints

import matplotlib.pyplot as plt

# %%
import numpy as np

from generalized_additive_models import GAM, Spline

# %%
num_samples = 100

rng = np.random.default_rng(42)
X = rng.triangular(-1, 0, 1, size=(num_samples, 1))
X_smooth = np.linspace(-1, 1, num=2**10).reshape(-1, 1)

y = np.sin(X.ravel() * 2.5) + rng.normal(size=(num_samples), scale=0.15) + 2

# %%
constraints = [
    None,
    "increasing",
    "convex",
    "concave",
    "increasing-concave",
    "increasing-convex",
]


fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True, sharey=True)
fig.suptitle("Regression with constraints")

for constraint, ax in zip(constraints, axes.ravel()):
    # Create a GAM
    spline = Spline(0, constraint=constraint, extrapolation="linear")
    gam = GAM(spline).fit(X, y)

    ax.set_title(str(constraint))
    ax.scatter(X, y, s=5, alpha=0.8, zorder=5)

    ax.plot(X_smooth, gam.predict(X_smooth), color="k", zorder=10, lw=2)
    ax.grid(True, ls="--", alpha=0.5, zorder=0)


fig.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True, sharey=False)
fig.suptitle("Basis functions")

for constraint, ax in zip(constraints, axes.ravel()):
    # Create a Spline
    spline = Spline(0, constraint=constraint, extrapolation="linear", num_splines=6)

    ax.set_title(str(constraint))
    ax.plot(X_smooth, spline.fit_transform(X_smooth))
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True, sharey=False)
fig.suptitle("Nullspace (functions with high penalty)")

num_functions = 10

for constraint, ax in zip(constraints, axes.ravel()):
    # Create a Spline
    spline = Spline(0, constraint=constraint, extrapolation="linear", num_splines=6)
    spline_basis = spline.fit_transform(X_smooth)

    # Draw some betas in the null space
    # They must be positive and linear (second derivative zero)
    beta = np.outer(
        np.linspace(0, 1, num=spline.num_coefficients),
        np.abs(rng.normal(size=num_functions)),
    )
    beta = beta + np.abs(rng.normal(size=(1, num_functions)))

    # Plot some functions drawn from the null space
    ax.set_title(str(constraint))
    ax.plot(X_smooth, spline_basis @ beta, color="black", alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
