"""
=================
Univariate spline
=================

Plot a univariate spline.

"""

import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models import GAM, Spline

# Create data
rng = np.random.default_rng(42)
x = np.linspace(-3, 3, num=2**6)
mu = np.sin(x)
y = np.sin(x) + rng.normal(loc=0, scale=0.33, size=len(x))
X = x[:, np.newaxis]

# Create a model
gam = GAM(Spline(0))
gam.fit(X, y)

# Predict on a smooth grid
x_smooth = np.linspace(-3.5, 3.5, num=2**10)
y_smooth = gam.predict(x_smooth[:, np.newaxis])

# Create a plot
plt.scatter(x, y, label="Data", color="black")
plt.plot(x_smooth, np.sin(x_smooth), label="Truth")
plt.plot(x_smooth, y_smooth, label="GAM")
plt.legend()
plt.show()
