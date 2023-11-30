"""
============
Spline basis
============

Plot spline bases.

"""
import matplotlib.pyplot as plt
import numpy as np

from generalized_additive_models import Spline

# Create data
rng = np.random.default_rng(42)
x = np.linspace(-3, 3, num=2**10)
X = x[:, np.newaxis]

# Create a Spline
spline = Spline(0, num_splines=10, degree=3)
X_transformed = spline.fit_transform(X)

# Transform by adding back the means, for pretty plotting
X_transformed = X_transformed + spline.means_

plt.title("Spline basis")
plt.plot(x, X_transformed)
plt.show()
