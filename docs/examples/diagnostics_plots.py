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
# ## Diagnostics plots
#
# Import packages.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

from generalized_additive_models import GAM, Categorical, Spline

# %% [markdown]
# Load data.

# %%
data = load_diabetes(as_frame=True)
df = data.data
y = data.target

# Code the "sex" variable as strings
df = df.assign(sex=lambda df: np.where(df.sex < 0, "Male", "Female"))

df.sample(5)

# %%
x = np.linspace(0, 2 * np.pi, num=2**10)
y = np.sin(x) + np.random.randn(len(x)) / 10
X = x.reshape(-1, 1)

gam = GAM(Spline(0)).fit(X, y)

# %% [markdown]
# ### Diagnostics with sklearn

# %%
from sklearn.metrics import PredictionErrorDisplay

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

ax1.set_title("Actual vs. predicted")
PredictionErrorDisplay.from_estimator(gam, X, y, ax=ax1, kind="actual_vs_predicted")
ax1.grid(True, ls="--", alpha=0.2)

ax2.set_title("Residuals vs. predicted")
PredictionErrorDisplay.from_estimator(gam, X, y, ax=ax2, kind="residual_vs_predicted")
ax2.grid(True, ls="--", alpha=0.2)

fig.tight_layout()
