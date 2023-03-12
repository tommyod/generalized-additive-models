.. -*- mode: rst -*-

|Actions|_ |PythonVersion|_ |PyPi|_ |Black|_

.. |Actions| image:: https://github.com/tommyod/generalized-additive-models/workflows/Python%20CI/badge.svg?branch=main
.. _Actions: https://github.com/tommyod/generalized-additive-models/actions/workflows/build.yml?query=branch%3Amain

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue
.. _PythonVersion: https://pypi.org/project/generalized-additive-models

.. |PyPi| image:: https://img.shields.io/pypi/v/generalized-additive-models
.. _PyPi: https://pypi.org/project/generalized-additive-models

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. image:: https://raw.githubusercontent.com/tommyod/generalized-additive-models/prepare-alpha/docs/_static/readme_figure.png?token=GHSAT0AAAAAABHJPRNESAGIMSDHQ652ZK74ZAKCICA
  :target: https://github.com/tommyod/generalized-additive-models/


generalized-additive-models
---------------------------

About
-----

Generalized Additive Models (GAM) are the `Predictive Modeling Silver Bullet <https://web.archive.org/web/20210812020305/https://multithreaded.stitchfix.com/assets/files/gam.pdf>`_.
A GAM is a statistical model in which the target variable depends on unknown smooth functions of the features, 
and interest focuses on inference about these smooth functions.

.. image:: https://latex.codecogs.com/svg.image?Y_i&space;\sim&space;\textup{ExponentialFamily}(\mu_i,&space;\phi)&space;\\g(\mu_i)&space;=&space;f_1(x_{i1})&space;&plus;&space;f_2(x_{i2})&space;&plus;&space;f_3(x_{i3},&space;x_{i4})&space;&plus;&space;\cdots
  
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

Citing
------

TODO
