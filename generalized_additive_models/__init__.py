#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:29:54 2023

@author: tommy
"""

__version__ = "0.0.0"


def grouper(keys):
    """Yield each term separately.

    Examples
    --------
    >>>
    """
    generator = iter(keys)

    to_yield = []
    prev_index = "0"
    for current in generator:
        index, delim, param = current.partition("__")
        if index != prev_index:
            yield to_yield
            prev_index = index
            to_yield = []

        to_yield.append(param)

    yield to_yield


keys = ["0", "0__by", "0__feature", "0__penalty", "1"]

print(list(grouper(keys)))
