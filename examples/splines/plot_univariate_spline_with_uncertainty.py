"""
==================================
Univariate spline with uncertainty
==================================

Plot a univariate spline with uncertainty.

"""
import matplotlib.pyplot as plt
import numpy as np

from generalized_additive_models import GAM, Spline

plt.title("Spline with uncertainty")

# Create more data on one end of the domain to show varying uncertainty
rng = np.random.default_rng(1)
x = -3 + 6 * np.sort(rng.triangular(left=0, mode=0, right=1, size=2**8) ** 2)
y = np.sin(x) + rng.normal(loc=0, scale=0.33, size=len(x))
X = x[:, np.newaxis]

# To evaluate the model on
x_smooth = np.linspace(-3.5, 3, num=2**10)
X_smooth = x_smooth[:, np.newaxis]

# Plot the data
plt.scatter(x, y, label="Data", color="black", s=5)

# Fit a model
spline = Spline(0, penalty=1e1)
gam = GAM(spline, fit_intercept=True)
gam.fit(X, y)

# Plot the mean value
y_smooth = gam.predict(X_smooth)
plt.plot(x_smooth, y_smooth, label="Mean")

# Transform the new grid
X_t = gam.terms.transform(X_smooth)

# Get the covariance matrix associated with the coefficients
V = gam.results_.covariance

# The variance of y = X @ \beta is given by diag(X @ V @ X.T)
stdev_array = np.sqrt(np.sum((X_t @ V) * X_t, axis=1))
upper = y_smooth + stdev_array * 3
lower = y_smooth - stdev_array * 3
plt.fill_between(x_smooth, lower, upper, zorder=5, alpha=0.33, label="±3 std")

# Create a plot
plt.legend()
plt.show()
