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
# ## Getting started
#
# Import packages.

# %%
# Import packages

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

from generalized_additive_models import GAM, Categorical, Intercept, Linear, Spline
from generalized_additive_models.inspection import partial_effect

# %% [markdown]
# Load data.

# %%
data = load_diabetes(as_frame=True)
df = data.data
y = data.target

# Code the "sex" variable as strings
df = df.assign(sex=lambda df: np.where(df.sex < 0, "Male", "Female"))

# %%
df

# %% [markdown]
# ### Creating a model
#
# Construct and fit a model.

# %%
# Features can be string and refer to column names if X is a pandas DataFrame
# If X was a 2D numpy array, features must instead be integers referring to the columns
num_splines = 12
penalty = 1
terms = (
     Spline("age", penalty=penalty, num_splines=num_splines)
     + Categorical("sex", penalty=1)
    + Spline("bmi", penalty=penalty, num_splines=num_splines)
    + Spline("bp", penalty=penalty, num_splines=num_splines)
)
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
# ### Partial effects and partial residuals
#
# Partial effects may be plotted as follows:

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 5))

# Loop over the individual terms in the model
for ax, term in zip(axes.ravel(), model.terms):
    # Skip plotting intercept terms and categorical terms
    if isinstance(term, (Intercept)):
        continue

    results = partial_effect(model, term, standard_deviations=1.0, linear_scale=True)

    # Create a plot
    ax.set_title(f"Partial effects for term: '{term.feature}'")

    # Linear and Spline terms
    if isinstance(term, (Linear, Spline)):
        
        ax.plot(results.x, results.y)
        ax.plot(results.x, results.y_low, color="k", ls="--")
        ax.plot(results.x, results.y_high, color="k", ls="--")

        # Rugplot
        minimum_value = np.min(results.y_low)
        ax.scatter(
            results.x_obs,
            np.ones_like(results.x_obs) * minimum_value,
            marker="|",
            color="black",
        )

        ax.grid(True, ls="--", alpha=0.2)

    # Categorical terms get a slightly different plot
    elif isinstance(term, Categorical):
        x_ticks = np.arange(len(results.x))
        x_labels = term.categories_
        
        ax.scatter(x_labels, results.y)
        ax.scatter(x_labels, results.y_low, color="k", ls="--")
        ax.scatter(x_labels, results.y_high, color="k", ls="--")

        ax.grid(True, ls="--", alpha=0.2)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)


fig.tight_layout()
plt.show()

# %% [markdown]
# We can also plot partial residuals.

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 5))

terms = [term for term in model.terms if isinstance(term, (Linear, Spline))]

# Loop over the individual terms in the model
for ax, term in zip(axes.ravel(), terms):
    results = partial_effect(model, term, standard_deviations=1.0, linear_scale=True)

    # Create a plot
    ax.set_title(f"Partial effects for term: '{term.feature}'")

    ax.plot(results.x, results.y)
    ax.plot(results.x, results.y_low, color="k", ls="--")
    ax.plot(results.x, results.y_high, color="k", ls="--")

    # Partial residuals
    minimum_value = np.min(results.y_low)
    ax.scatter(results.x_obs, results.y_partial_residuals, color="black", s=1, alpha=0.33)

    ax.grid(True, ls="--", alpha=0.2)


fig.tight_layout()
plt.show()

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

# %% [markdown]
# ### Constraints

# %%
plt.figure(figsize=(6, 3))
plt.scatter(df["bp"], y, s=2, alpha=0.5, color="k")

# Create smooth grid to evaluate on
X_smooth = np.linspace(df["bp"].min(), df["bp"].max(), num=2**10).reshape(-1, 1)

# Loop over constraints and create a model to fit and plot
for constraint in [None, "increasing", "increasing-concave"]:
    model = GAM(Spline("bp", constraint=constraint)).fit(df, y)

    plt.plot(X_smooth, model.predict(X_smooth), label=str(constraint))

plt.legend()
plt.tight_layout()
plt.show()
