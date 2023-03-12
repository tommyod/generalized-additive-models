#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:01:43 2023

@author: tommy
"""

from sklearn.base import clone

from generalized_additive_models import Normal


class TestSklearnCompatibility:
    def test_cloning_with_sklearn_clone(self):
        normal = Normal(10)
        cloned = clone(normal)
        normal.scale = 1

        assert cloned.scale == 10

    def test_that_get_and_set_params_works(self):
        normal = Normal(13)
        assert normal.get_params(True) == {"scale": 13}
        assert normal.get_params(False) == {"scale": 13}

        normal.set_params(scale=1)
        assert normal.get_params() == {"scale": 1}


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
