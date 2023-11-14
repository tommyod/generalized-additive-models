.. -*- mode: rst -*-

|Actions|_ |PythonVersion|_ |PyPi|_ |Black|_ |ReadtheDocs|_

.. |Actions| image:: https://github.com/tommyod/generalized-additive-models/workflows/Python%20CI/badge.svg?branch=main
.. _Actions: https://github.com/tommyod/generalized-additive-models/actions/workflows/build.yml?query=branch%3Amain

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11|%203.12%20-blue
.. _PythonVersion: https://pypi.org/project/generalized-additive-models

.. |PyPi| image:: https://img.shields.io/pypi/v/generalized-additive-models
.. _PyPi: https://pypi.org/project/generalized-additive-models

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |Downloads| image:: https://pepy.tech/badge/generalized-additive-models
.. _Downloads: https://pepy.tech/project/generalized-additive-models

.. |ReadtheDocs| image:: https://readthedocs.org/projects/generalized-additive-models/badge/
.. _ReadtheDocs: https://generalized-additive-models.readthedocs.io/en/latest/


generalized-additive-models
===========================

Generalized Additive Models (GAMs) in Python.

About
-----

GAMs are uniquely placed on the interpretability vs. precitive power continuum.
In many applications they perform almost as well as more complex models, but are extremely interpretable.

- GAMs extend linear regression by allowing non-linear relationships between features and the target.
- The model is still additive, but link functions and multivariate splines facilitate a broad class of models.
- While GAMs are likely outperformed by non-additive models (e.g. boosted trees), GAMs are extremely interpretable.

Read more about GAMs:

- `Predictive Modeling Silver Bullet <https://web.archive.org/web/20210812020305/https://multithreaded.stitchfix.com/assets/files/gam.pdf>`_
- `Generalized Additive Models: An Introduction with R <https://www.amazon.com/Generalized-Additive-Models-Introduction-Statistical/dp/1498728332>`_

A GAM is a statistical model in which the target variable depends on unknown smooth functions of the features, 
and interest focuses on inference about these smooth functions.
  
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
    

Go to `Read the Docs <https://generalized-additive-models.readthedocs.io/en/latest/>`_ to see full documentation.

Contributing and development
----------------------------

Contributions are very welcome.
You can correct spelling mistakes, write documentation, clean up code, implement new features, etc.

Some guidelines for development:

- Code must comply with the standard. See the GitHub action pipeline for more information.
- If possible, use existing algorithms from `numpy`, `scipy` and `scikit-learn`.
- Write tests, especically regression tests if a bug is fixed.
- Take backward compatibility seriously. API changes require good reason.

Installation for local development::

    pip install -e '.[dev,lint,doc]'
    
Create documentation locally::

    sudo apt install pandoc
    sphinx-build docs _built_docs/html -W -a -E --keep-going
    sphinx-autobuild docs _built_docs/html -v -j "auto" --watch generalized_additive_models
    
Once the `version` has been incremented, the commit must be tagged and pushed in order to publish to PyPi::

    git tag -a v0.1.0 -m "Version 0.1.0" b22724c
    git push origin v0.1.0

Citing
------

TODO
