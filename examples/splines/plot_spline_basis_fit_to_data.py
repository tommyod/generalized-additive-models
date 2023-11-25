"""
========================
Spline basis fit to data
========================

Plot a spline basis and its fit to data.

"""
import matplotlib.pyplot as plt
import numpy as np

from generalized_additive_models import GAM, Spline

# Create data
rng = np.random.default_rng(42)
x = np.linspace(-3, 3, num=2**6)
y = 2 + np.sin(x) + rng.normal(loc=0, scale=0.33, size=len(x))
X = x[:, np.newaxis]

# Plot data
plt.scatter(x, y, color="black")

# Create a GAM and fit it
spline = Spline(0, num_splines=12, degree=3)
gam = GAM(spline).fit(X, y)

# Plot the fitted GAM
x_smooth = np.linspace(-3, 3, num=2**10)
y_smooth = gam.predict(x_smooth[:, np.newaxis])
plt.plot(x_smooth, y_smooth)

# Plot the Splines, multiplied by coefficients
X_transformed = spline.transform(x_smooth[:, np.newaxis])
X_times_coef = X_transformed * spline.coef_
plt.plot(x_smooth, X_times_coef)
plt.show()
