"""
=======================
Plot Poisson regression
=======================

Plot a Poisson regression on a time series dataset.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_poisson_deviance
from sklearn.model_selection import GridSearchCV, KFold

from generalized_additive_models import GAM, Spline
from generalized_additive_models.datasets import load_bicycles

# Load data and filter it
df = load_bicycles()
df = df.loc[lambda df: df.station_name == "Hillevåg", ["date", "count"]]
df = df.assign(date=lambda df: pd.to_datetime(df.date))

# Group by week, choose a single year and get week numbers
df = df.set_index("date").resample("W").sum().reset_index()
df = df[df.date.dt.isocalendar().year == 2019]
df = df.assign(weeknumber=lambda df: df.date.dt.isocalendar().week.values)

# Create periodic spline model
terms = Spline("weeknumber", penalty=1e4, num_splines=32, extrapolation="periodic")
gam = GAM(terms=terms, distribution="poisson", link="log")
gam.fit(df, df["count"])

# Grid search for optimal value of the penalty
cv = KFold(shuffle=True, random_state=42, n_splits=5)
scoring = make_scorer(mean_poisson_deviance, greater_is_better=False)
grid_search = GridSearchCV(
    gam, {"terms__0__penalty": np.logspace(1, 3, num=8)}, cv=cv, scoring=scoring
)
grid_search.fit(df, df["count"])
print("Optimal parameters:", grid_search.best_params_)
gam = grid_search.best_estimator_

# To evaluate the model on
x_smooth = np.linspace(1, 53, num=2**10)

# Plot the data and predictions
plt.scatter(df["weeknumber"], df["count"], label="Data", color="black", s=8)
plt.plot(x_smooth, gam.predict(x_smooth[:, None]), label="Poisson spline")

# Create a plot
plt.xlabel("Week number")
plt.ylabel("Number of bicycles")
plt.grid(True, ls="--", alpha=0.33)
plt.legend()
plt.show()
