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
# ## One dimensional regression
#
# This example shows how to create a one dimensional regression.
#
# ### Imports and data creation

import matplotlib.pyplot as plt

# %%
import numpy as np

from generalized_additive_models import GAM, Spline

rng = np.random.default_rng(42)
X = rng.triangular(left=-1, mode=0, right=1, size=(100, 1))
y = (np.sin(X * 2.5) + 1 + rng.normal(size=(100, 1), scale=0.3)).ravel()

# %% [markdown]
# ### Fit a simple GAM

# %%
# Create a model and fit it
gam = GAM(Spline(0, extrapolation="continue"))
gam.fit(X, y)

# Create a smooth grid to predict on
X_smooth = np.linspace(-1, 1, num=2**10).reshape(-1, 1)

plt.figure(figsize=(8, 3))
plt.scatter(X, y, s=5)
plt.plot(X_smooth, gam.predict(X_smooth), color="k")
plt.grid(True, ls="--", alpha=0.33, zorder=0)
plt.show()

# %% [markdown]
# ### Demonstrate the effect of penalization

# %%
penalties = [0.1, 1, 100, 10000]

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 5))
fig.suptitle("Effect of penalty (higher penalty -> less wiggliness)")

for penalty, ax in zip(penalties, axes.ravel()):
    spline = Spline(0, penalty=penalty, extrapolation="continue")
    gam = GAM(spline).fit(X, y)

    ax.set_title(f"Penalty={penalty}")
    ax.scatter(X, y, s=5)
    ax.plot(X_smooth, gam.predict(X_smooth), color="k")
    ax.grid(True, ls="--", alpha=0.33, zorder=0)
    ax.set_ylim([-1, 3])

fig.tight_layout()
