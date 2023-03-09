.. generalized-additive-models documentation master file, created by
   sphinx-quickstart on Sat Feb 18 21:33:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to generalized-additive-models's documentation!
=======================================================

.. currentmodule:: generalized_additive_models



TODO: ABOUT

Installation
------------

.. code-block:: text

   $ pip install geneeralized-additive-models



TODO

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


Citing
------

   
Examples
========
   
.. toctree::
   :maxdepth: 1
   
   examples/getting_started
   API documentation <API>
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
