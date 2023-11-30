"""
====================================
Univariate spline with extrapolation
====================================

Plot a univariate spline with extrapolation.

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

# To evaluate the model on
x_smooth = np.linspace(-3, 7.5, num=2**10)

# Plot the data
plt.scatter(x, y, label="Data", color="black")

for extrapolation in ["continue", "linear", "constant", "periodic"]:
    # Create a model
    gam = GAM(Spline(0, extrapolation=extrapolation))
    gam.fit(X, y)

    # Predict on a smooth grid
    y_smooth = gam.predict(x_smooth[:, np.newaxis])
    plt.plot(x_smooth, y_smooth, label=f"Extrapolation={extrapolation}")

# Create a plot
plt.legend()
plt.show()
