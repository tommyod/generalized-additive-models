"""
===================================
Plot categorical terms with Tensors
===================================

We examine relative strength on the powerlifting dataset across exercises.

"""
import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models import GAM, Categorical, Spline
from generalized_additive_models.datasets import load_powerlifters

rng = np.random.default_rng(42)

# Load data and filter it
df = load_powerlifters().rename(
    columns=lambda s: s.removeprefix("best3").removesuffix("kg")
)

# Model terms
age = Spline("age")
bodyweight = Spline("bodyweight")
sex = Categorical("sex", penalty=0)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

# Loop over each exercise: squat, bench, deadlift
ex_names = ["squat", "bench", "deadlift"]
for ax, ex_name in zip(axes.ravel(), ex_names):
    # Create and fit model, adjusting for age and bodyweight
    terms = age + sex + bodyweight
    gam = GAM(terms=terms, distribution="normal", fit_intercept=False)
    gam.fit(df, df[ex_name])
    print(f"Explained variance ({ex_name}):", gam.score(df, df[ex_name]))

    # Relative effect
    coefs = sex.coef_ / sex.coef_[0]

    # Create plot
    ax.set_title(ex_name.capitalize())
    p = ax.bar(np.arange(2), coefs)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(sex.categories_)
    ax.bar_label(p, coefs.round(2), label_type="center")
    ax.grid(True, ls="--", alpha=0.33)
    if ex_name == "squat":  # Only label first one
        ax.set_ylabel("Relative strength")

plt.tight_layout()
plt.show()
