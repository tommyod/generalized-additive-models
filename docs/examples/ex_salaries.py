# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Example: salaries
#
# Import packages.

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from generalized_additive_models import GAM, Categorical, Spline

# %% [markdown]
# Load data.

# %%
from generalized_additive_models.datasets import load_salaries

df = load_salaries()
y = df["salary"].values
df.sample(5, random_state=42)

# %% [markdown]
# ### Create sklearn pipeline

# %%
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


numerical_features = [
    "age",
    "years_formal_relevant_education",
    "years_relevant_work_experience",
]

categorical_features = [
    "sector",
    "geographical_location",
    "work_domain",
    "num_colleagues",
]


pipe_union = ColumnTransformer(
    [
        ("numerical", "passthrough", numerical_features),
        ("categorical", OneHotEncoder(), categorical_features),
    ]
)

pipeline = Pipeline([("transform", pipe_union), ("model", DummyRegressor())])

pipeline

# %% [markdown]
# ### Model: dummy

# %%
from sklearn.model_selection import cross_validate

pipeline.steps[-1] = ("model", DummyRegressor())

results_dummy = cross_validate(pipeline, df, y, cv=10, scoring="r2")

(results_dummy["test_score"].mean(), results_dummy["test_score"].std())

# %% [markdown]
# ### Model: linear

# %%
from sklearn.linear_model import Ridge

FEATURES = ["years_formal_relevant_education", "years_relevant_work_experience"]

pipeline.steps[-1] = ("model", Ridge())

results_linear = cross_validate(pipeline, df, y, cv=10, scoring="r2")

(results_linear["test_score"].mean(), results_linear["test_score"].std())

# %% [markdown]
# ### Model: boosting

# %%
from sklearn.ensemble import GradientBoostingRegressor

pipeline.steps[-1] = ("model", GradientBoostingRegressor())

results_boosting = cross_validate(pipeline, df, y, cv=10, scoring="r2")

(results_boosting["test_score"].mean(), results_boosting["test_score"].std())

# %% [markdown]
# ### Create a GAM

# %%
penalty_spline = 10
penalty_categorical = 5

terms = (
    Spline("age", penalty=penalty_spline, num_splines=8)
    + Spline("years_formal_relevant_education", penalty=penalty_spline, num_splines=8)
    + Spline("years_relevant_work_experience", penalty=penalty_spline, num_splines=8)
    + Categorical("sector", penalty=penalty_categorical)
    + Categorical("geographical_location", penalty=penalty_categorical)
    + Categorical("work_domain", penalty=penalty_categorical)
)

model = GAM(terms, link="identity")

results_gam = cross_validate(model, df, y, cv=10, scoring="r2")

(results_gam["test_score"].mean(), results_gam["test_score"].std())

# %%
model.fit(df, y)
model.summary()

# %% [markdown]
# ### Inspect overall fit

# %%
from generalized_additive_models.inspection import QQDisplay, ResidualScatterDisplay

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

QQDisplay.from_estimator(model, df, y, ax=ax1)
ResidualScatterDisplay.from_estimator(model, df, y, ax=ax2)

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Inspect partial effects for spline features

# %%
from generalized_additive_models.inspection import PartialEffectDisplay


for term in model.terms:
    if not isinstance(term, Spline):
        continue

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.set_title(term.feature)
    PartialEffectDisplay.from_estimator(model, term, df, y, ax=ax, rug=False)
    ax.grid(True, ls="--", zorder=0, alpha=0.33)
    plt.show()

# %% [markdown]
# ### Inspect categorical features

# %%
for term in model.terms:
    if not isinstance(term, Categorical):
        continue

    # Collect the information in a Series
    series = pd.Series(term.coef_, index=term.categories_).sort_values()

    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(6, 1 + 0.2 * len(series)))
    ax.set_title(term.feature)

    ticks = np.arange(len(series))
    ax.barh(ticks, series.values)
    ax.set_yticks(ticks)
    ax.set_yticklabels(series.index)
    ax.grid(True, ls="--", zorder=0, alpha=0.33)

    fig.tight_layout()
    plt.show()
