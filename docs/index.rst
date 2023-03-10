.. generalized-additive-models documentation master file, created by
   sphinx-quickstart on Sat Feb 18 21:33:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to generalized-additive-models's documentation!
=======================================================

.. currentmodule:: generalized_additive_models

generalized-additive-models
---------------------------

About
-----

Generalized Additive Models (GAM) are the `Predictive Modeling Silver Bullet <https://web.archive.org/web/20210812020305/https://multithreaded.stitchfix.com/assets/files/gam.pdf>`_.
A GAM is a statistical model in which the target variable depends on unknown smooth functions of the features, 
and interest focuses on inference about these smooth functions.


.. math::

   Y_i &\sim \textup{ExponentialFamily}(\mu_i, \phi) \\
   g(\mu_i) &= f_1(x_{i1}) + f_2(x_{i2}) + f_3(x_{i3}, x_{i4}) + \cdots

An exponential family distribution is specified for the target Y (.e.g Normal, Binomial or Poisson) 
along with a link function g (for example the identity or log functions) relating the 
expected value of Y to the predictor variables.


Installation
------------

Install using pip::

    pip install generalized-additive-models


Example
-------

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import cross_val_score
    from generalized_additive_models import GAM, Spline, Categorical
        
    # Load data
    data = load_diabetes(as_frame=True)
    df, y = data.data, data.target
    
    # Create model
    terms = Spline("bp") + Spline("bmi", constraint="increasing") + Categorical("sex")
    gam = GAM(terms)
    
    # Cross validate
    scores = cross_val_score(gam, df, y, scoring="r2")
    print(scores) # array([0.26, 0.4 , 0.41, 0.35, 0.42])

Contributing
------------

Contributions are very welcome.
You can correct spelling mistakes, write documentation, clean up code, implement new features, etc.

Some guidelines:

- Code must comply with the standard. See the GitHub action pipeline for more information.
- If possible, use existing algorithms from `numpy`, `scipy` and `scikit-learn`.
- Write tests, especically regression tests if a bug is fixed.
- Take backward compatibility seriously. API changes require good reason.

   
Examples
========
   
.. toctree::
   :maxdepth: 1
   
   examples/getting_started
   API documentation <API>
   references
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
