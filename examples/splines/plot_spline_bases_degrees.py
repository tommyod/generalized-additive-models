"""
===============================
Spline bases of varying degrees
===============================

Plot spline bases of varying degrees.

"""
import matplotlib.pyplot as plt
import numpy as np

from generalized_additive_models import Spline

# Create data
rng = np.random.default_rng(42)
x = np.linspace(-3, 3, num=2**10)
X = x[:, np.newaxis]

# Create a 2x2 plot and loop over each axis
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

for degree, ax in enumerate(axes.ravel()):
    ax.set_title(f"Spline basis (degree {degree})")

    # Create a Spline
    spline = Spline(0, num_splines=6, degree=degree)
    X_transformed = spline.fit_transform(X)

    # Transform by adding back the means, for pretty plotting
    X_transformed = X_transformed + spline.means_
    ax.plot(x, X_transformed, alpha=0.8)

plt.tight_layout()
plt.show()
