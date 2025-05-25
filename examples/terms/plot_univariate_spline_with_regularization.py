"""
=====================================
Univariate spline with regularization
=====================================

Plot a univariate spline with regularization.

"""

import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models import GAM, Spline
from generalized_additive_models.datasets import load_mcycle

# Create data
df = load_mcycle()
X, y = df[["times"]], df["accel"]

# To evaluate the model on
x_smooth = np.linspace(2, 65, num=2**10)

# Plot the data
plt.scatter(X, y, label="Data", color="black", s=8)

for penalty in [10**4, 10**2, 1, 0.001]:
    # Create a model
    gam = GAM(Spline("times", penalty=penalty))
    gam.fit(X, y)

    # Predict on a smooth grid
    y_smooth = gam.predict(x_smooth[:, np.newaxis])
    plt.plot(x_smooth, y_smooth, label=f"Penalty={penalty}")

# Create a plot
plt.grid(True, ls="--", alpha=0.33)
plt.legend()
plt.show()
