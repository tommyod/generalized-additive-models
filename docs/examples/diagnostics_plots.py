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

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ### Creating a model
#
# Construct and fit a model.

# %%
# Features can be string and refer to column names if X is a pandas DataFrame
# If X was a 2D numpy array, features must instead be integers referring to the columns

terms = Spline("age") + Categorical("sex", penalty=1) + Spline("bmi") + Spline("bp")
model = GAM(terms, link="identity", verbose=2, fit_intercept=False)

# Fit the model
model.fit(df, y)

# %%
model.summary()

# %% [markdown]
# Make a prediction and score the model.

# %%
predictions = model.predict(df)
model.score(df, y)  # Pseudo R2 score

# %% [markdown]
# Since this model has a normal distribution and an identity link, the score equal the R2 score.

# %%
from sklearn.metrics import r2_score

r2_score(y_true=y, y_pred=predictions)

# %% [markdown]
# ### Scikit-learn compatibility
#
# The models interact nicely with scikit-learn.

# %%
from sklearn.model_selection import cross_val_score

# Models can be used with sklearn's `cross_val_score`
model = GAM(Spline("bp"))
cross_val_score(model, df, y, scoring="r2")

# %%
from sklearn.model_selection import GridSearchCV

model = GAM(Spline("bp"))

# A parameter grid can be used to search for e.g. the best spline penalty
param_grid = {"terms__penalty": [1, 10, 100, 1000]}
search = GridSearchCV(model, param_grid, scoring="r2").fit(df, y)
search.best_params_

# %%
# To search over more than one term, use a `param_grid` like this
model = GAM(Spline("bp") + Spline("age"))

# The integers are used to index the terms, so 0 refers to the first term, etc
param_grid = {
    "terms__0__penalty": [1, 10, 100, 1000],
    "terms__1__penalty": [1, 10, 100, 1000],
}
search = GridSearchCV(model, param_grid, scoring="r2").fit(df, y)
search.best_params_

# %% [markdown]
# ### Penalties

# %%
plt.figure(figsize=(6, 3))
plt.scatter(df["bp"], y, s=2, alpha=0.5, color="k")

X_smooth = np.linspace(df["bp"].min(), df["bp"].max(), num=2**10).reshape(-1, 1)

for penalty in [0.001, 1, 1000]:
    model = GAM(Spline("bp", penalty=penalty)).fit(df, y)

    plt.plot(X_smooth, model.predict(X_smooth), label=f"Penalty={penalty}")

plt.legend()
plt.tight_layout()
plt.show()
