"""
======================
Plot categorical terms
======================

We examine relative strength on the powerlifting dataset.

"""
import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models import GAM, Categorical, Spline
from generalized_additive_models.datasets import load_powerlifters

# Load data and filter it
df = load_powerlifters()

# Predict total weight lifted, given age, bodyweight and sex
# Large penalties because of the log-link
target = df["totalkg"]
age = Spline("age", penalty=1e6, num_splines=8)
bodyweight = Spline("bodyweightkg", penalty=1e6, num_splines=8)
sex = Categorical("sex", penalty=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

# ============== MODEL: strength ~ spline(age) + categorical(sex)
ax1.set_title("Relative strength\n(adjusting for age)")

terms = age + sex
gam = GAM(terms=terms, distribution="normal", link="log", fit_intercept=False)
gam.fit(df, target)
print("Explained variance (no bodyweight):", gam.score(df, target))

# We used a log-link, so the predicted strength will be exp() of coefs.
# We also normalize the categories to get a relative number
coefs = np.exp(sex.coef_) / np.exp(sex.coef_[0])

p = ax1.bar(np.arange(2), coefs)
ax1.set_xticks(np.arange(2))
ax1.set_xticklabels(sex.categories_)
ax1.bar_label(p, coefs.round(2), label_type="center")
ax1.grid(True, ls="--", alpha=0.33)
ax1.set_ylabel("Relative strength")

# ============== MODEL: strength ~ spline(age) + categorical(sex)
ax2.set_title("Relative strength\n(adjusting for age and bodyweight)")

terms = age + sex + bodyweight  # Same as above, but with bodyweight too
gam = GAM(terms=terms, distribution="normal", link="log", fit_intercept=False)
gam.fit(df, target)
print("Explained variance (with bodyweight):", gam.score(df, target))

coefs = np.exp(sex.coef_) / np.exp(sex.coef_[0])

p = ax2.bar(np.arange(2), coefs)
ax2.set_xticks(np.arange(2))
ax2.set_xticklabels(sex.categories_)
ax2.bar_label(p, coefs.round(2), label_type="center")
ax2.grid(True, ls="--", alpha=0.33)
ax2.set_ylabel("Relative strength")

plt.show()
