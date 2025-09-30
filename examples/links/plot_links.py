"""
==================
Plot distributions
==================

Plot some of the distributions available.

"""

import matplotlib.pyplot as plt
import numpy as np
from generalized_additive_models.links import LINKS

fig, axes = plt.subplots(2, 4, figsize=(7, 3.5))
axes = axes.ravel()


for ax, (link_name, link_func) in zip(axes, LINKS.items()):
    ax.set_title(link_name)

    # The inverse link functions maps from an
    # unbounded linear space to the domain
    x = np.linspace(-5, 5)
    ax.plot(x, link_func().inverse_link(x))
    ax.grid(True, ls="--", alpha=0.33)

plt.tight_layout()
plt.show()
