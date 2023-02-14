#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:29:54 2023

@author: tommy
"""

__version__ = "0.0.0"


import logging

# This goes into your library somewhere
logging.getLogger("generalized_additive_models").addHandler(logging.NullHandler())


from generalized_additive_models.gam import GAM
from generalized_additive_models.links import Identity
