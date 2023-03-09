API
---

.. currentmodule:: generalized_additive_models

This page contains API documentation for the classes.
   
   
Terms
.....

Terms are used in GAMs to model the features.
For instance, we can use :py:class:`Spline` and :py:class:`Categorical` as follows:

.. code-block:: python

   terms = Spline("age") + Categorical("sex")
   
This constructs a :py:class:`TermList`, which can be passed to a :py:class:`GAM`.

.. code-block:: python

   model = GAM(terms)

Here are all the available terms:

   
.. autosummary::
   :toctree: API

   Intercept
   Linear
   Categorical
   Spline
   Tensor
   
TermList
........

A :py:class:`TermList` is a subclass of :py:class:`list`, designed to hold terms.
   
.. autosummary::
   :toctree: API

   TermList


Links
.....

Link functions relate linear predictions :math:`\eta_i = X_i \beta` to the expected values :math:`\mu_i` of an exponential family distribution.

.. math::

   y_i &\sim \operatorname{ExpFam}(\mu_i, \phi) \\
   g(\mu_i) &= \eta_i = X_i \beta
   
For instance, if you believe the features multiply together to create :math:`\mu_i`, you can model this with the :py:class:`Log` link as 

.. math::

   y_i &\sim \operatorname{Normal}(\mu_i, \phi) \\
   \log(\mu_i) & = X_i \beta
   
This implies that :math:`\mu_i = \exp(X_i \beta) = \exp(X_{ij} \beta_j)`
   
.. autosummary::
   :toctree: API

   Identity
   Log
   Logit
   Softplus


Distributions
.............
   
.. autosummary::
   :toctree: API

   Normal
   Poisson
   Binomial


Models
......
   
.. autosummary::
   :toctree: API

   GAM
   ExpectileGAM