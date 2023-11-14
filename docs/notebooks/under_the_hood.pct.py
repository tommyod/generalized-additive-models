# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Under the hood
#
# Import packages.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models.datasets import load_salaries

from generalized_additive_models import GAM, Categorical, Spline, Intercept, Linear

rng = np.random.default_rng(42)

# %% [markdown]
# Load data.

# %%
df = load_salaries().sort_values("salary")
df[["age", "work_domain", "salary"]].sample(5, random_state=42)

# %% [markdown]
# ### Terms are scikit-learn Transformers
#
# Analogous to Transformers scikit-learn, a term such as `Linear` implement `fit` and `transform`.
#
# Below the `age` column is transformed by subtracting the mean value.

# %%
Linear("age").fit_transform(df)

# %% [markdown]
# Categorical variables are one-hot encoded.

# %%
Categorical("work_domain").fit_transform(df)

# %% [markdown]
# A spline basis is created when we fit and transform using a Spline.
# Under the hood, the implementation relies on the scikit-learn [SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html).
# For the visualization to look correct, we must sort the values.

# %%
X = np.sort(rng.triangular(left=-1, mode=0, right=1, size=2**10))[:, np.newaxis]
plt.figure(figsize=(8, 3))
plt.plot(X.ravel(), Spline(0, num_splines=6).fit_transform(X))
plt.show()

# %% [markdown]
# Every spline basis is centered, so the mean value over all data points equals zero.

# %%
Spline(0, num_splines=6).fit_transform(X).mean(axis=0)

# %%
Spline(0, num_splines=6).fit_transform(X).mean()

# %% [markdown]
# ### Terms may be added
#
# When we add two Terms, such as `Intercept` and `Linear`, a `TermList` instance is created.

# %%
terms = Intercept() + Linear("age")
terms

# %% [markdown]
# Fitting and transforming a `TermList` involves fitting each `Term` in the `TermList`.

# %%
terms.fit_transform(df)

# %% [markdown]
# We can access useful properties, used by the GAM:

# %%
print("Number of coefficients:", terms.num_coefficients)
print("Penalty matrix:\n", terms.penalty_matrix())

# %% [markdown]
# ### Penalties (regularization)

# %% [markdown]
# The penalty matrix $D$ goes into the regulatization term as
#
# $$\text{loss}(\beta)
# = \text{data fit} + \text{regularization}
# = \text{data fit} + \text{penalty} \lVert D \beta \rVert_2^2$$

# %%
D = (Intercept() + Linear("age", penalty=2)).penalty_matrix()
D

# %% [markdown]
# The penalty for a `Spline` acts on the second-order derivative (smoothness) of the spline.

# %%
Spline(0, num_splines=8).penalty_matrix()

# %% [markdown]
# A spline with no smoothness (affine function) results in no penalty.

# %%
spline = Spline(0, num_splines=6)
beta = np.arange(6)

spline.penalty_matrix() @ beta, np.linalg.norm(spline.penalty_matrix() @ beta)

# %%
plt.figure(figsize=(8, 3))
plt.plot(X.ravel(), spline.fit_transform(X) @ beta)
plt.show()

# %% [markdown]
# A wiggly, non-smooth spline does incur a penalty.

# %%
beta = np.array([1, 2, 1, 2, 1, 2])
spline.penalty_matrix() @ beta, np.linalg.norm(spline.penalty_matrix() @ beta)

# %%
plt.figure(figsize=(8, 3))
plt.plot(X.ravel(), spline.fit_transform(X) @ beta)
plt.show()

# %% [markdown]
# ### Univariate GAM

# %%
model = GAM(Spline("years_relevant_work_experience"), verbose=9)
model.fit(df, df["salary"])

# %%
years_smooth = np.linspace(0, 50, num=2**10)

plt.figure(figsize=(8, 3))
plt.scatter(df["years_relevant_work_experience"], df["salary"], alpha=0.33)

# Predict with the model
plt.plot(years_smooth, model.predict(years_smooth[:, np.newaxis]), color="black")

plt.show()

# %%
from sklearn.linear_model import Ridge

spline = Spline("years_relevant_work_experience")
X = spline.fit_transform(df)
y = df["salary"]

ridge = Ridge().fit(X, y)

plt.figure(figsize=(8, 3))
plt.scatter(df["years_relevant_work_experience"], df["salary"], alpha=0.33)
plt.plot(
    years_smooth,
    ridge.predict(spline.transform(years_smooth[:, np.newaxis])),
    color="black",
)
plt.plot
plt.show()

# %%
