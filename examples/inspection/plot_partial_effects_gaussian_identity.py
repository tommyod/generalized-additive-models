"""
=======================================
PartialEffectDisplay with identity link
=======================================

Plot a Gaussian regression on a dataset with powerlifters.

"""

import matplotlib.pyplot as plt
from generalized_additive_models import GAM, Categorical, Spline
from generalized_additive_models.datasets import load_powerlifters
from generalized_additive_models.inspection import PartialEffectDisplay

# Load data and filter it
df = load_powerlifters()

# Predict total weight lifted, given age, bodyweight and sex
target = df["totalkg"]
age = Spline("age", penalty=1e2)
bodyweight = Spline("bodyweightkg", penalty=1e2)
sex = Categorical("sex")

terms = age + bodyweight + sex
gam = GAM(terms=terms, distribution="normal", link="identity")
gam.fit(df, target)

print("Explained variance:", gam.score(df, target))

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

for term, ax in zip([age, bodyweight], axes.ravel()):
    ax.set_title(term.feature)

    # Use a subsample for the partial residuals
    df_subsample = df.sample(500, random_state=42)
    PartialEffectDisplay.from_estimator(
        gam,
        term,
        df_subsample,
        df_subsample["totalkg"],
        ax=ax,
        residuals=True,  # Plot partial residuals
        standard_deviations=3.0,  # Number of standard deviations
    )
    ax.grid(True, ls="--", alpha=0.33)

plt.tight_layout()
plt.show()
