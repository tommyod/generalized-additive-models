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

import matplotlib.pyplot as plt

# %%
import numpy as np

from generalized_additive_models import GAM, Spline

# %%
rng = np.random.default_rng(42)
X = rng.normal(size=(100, 1))
y = (np.sin(X) + 1 + rng.normal(size=(100, 1), scale=0.1)).ravel()

# %%
gam = GAM(Spline(0, penalty=10, extrapolation="continue"))
gam.fit(X, y)

plt.scatter(X, y, s=5)

X_smooth = np.linspace(-3, 3, num=2**10).reshape(-1, 1)

plt.plot(X_smooth, gam.predict(X_smooth), color="k")

plt.show()
