#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 08:25:44 2023

@author: tommy
"""


import io
import itertools

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy as sp
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.utils import resample

from generalized_additive_models.distributions import Binomial, Normal
from generalized_additive_models.gam import GAM, ExpectileGAM
from generalized_additive_models.links import Identity, Log, Logit
from generalized_additive_models.terms import Categorical, Intercept, Linear, Spline, Tensor, TermList


class TestAgainstRLM:
    def test_that_inference_of_factors_equals_Rs_lm_function(self):
        """

        height <- c(2.5, 1, 1.5, 1, 1, 0, 1, 0)
        gender <- factor(c("male","male","male","male","female","female","female", "female"))
        country <- factor(c("no", "us", "no", "us", "no", "us", "no", "us"))

        data <- data.frame(height, gender, country)

        model = lm(height ~ gender + country - 1, data=data)
        summary(model)

        ----------------------------
        Call:
        lm(formula = height ~ gender + country - 1, data = data)

        Residuals:
                 1          2          3          4          5          6          7          8
         5.000e-01 -1.943e-16 -5.000e-01  1.156e-16 -5.551e-17  2.776e-17 -5.551e-17  2.776e-17

        Coefficients:
                     Estimate Std. Error t value Pr(>|t|)
        genderfemale   1.0000     0.1936   5.164 0.003573 **
        gendermale     2.0000     0.1936  10.328 0.000146 ***
        countryus     -1.0000     0.2236  -4.472 0.006566 **
        ---
        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

        Residual standard error: 0.3162 on 5 degrees of freedom
        Multiple R-squared:   0.96,	Adjusted R-squared:  0.936
        F-statistic:    40 on 3 and 5 DF,  p-value: 0.0006425

        """

        df = pd.DataFrame(
            {
                "height": (2.5, 1, 1.5, 1, 1, 0, 1, 0),
                "gender": ("male", "male", "male", "male", "female", "female", "female", "female"),
                "country": ("no", "us", "no", "us", "no", "us", "no", "us"),
            }
        )

        gender_cat = Categorical("gender", penalty=0)
        country_cat = Categorical("country", penalty=0)

        gam = GAM(gender_cat + country_cat, fit_intercept=False).fit(df, df.height)

        # Standard errors are bounded
        assert np.all(np.sqrt(np.diag(gender_cat.coef_covar_)) < 0.23)
        assert np.all(np.sqrt(np.diag(country_cat.coef_covar_)) < 0.23)

        # Residuals match R results
        residuals = (df.height - gam.predict(df)).values
        residuals_R = np.array(
            [5.000e-01, -1.943e-16, -5.000e-01, 1.156e-16, -5.551e-17, 2.776e-17, -5.551e-17, 2.776e-17]
        )
        assert np.allclose(residuals, residuals_R)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            "--capture=sys",
            "--doctest-modules",
            "--maxfail=1",
        ]
    )
