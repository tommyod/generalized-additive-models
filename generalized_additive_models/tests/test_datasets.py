#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 07:42:05 2023

@author: tommy
"""

from generalized_additive_models.datasets import load_salaries


def test_load_salaries():
    df = load_salaries()
    assert df.shape == (2232, 9)
    assert df.isnull().sum().sum() == 0
