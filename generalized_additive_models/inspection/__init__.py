#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from generalized_additive_models.inspection.partial_effect import PartialEffectDisplay
from generalized_additive_models.inspection.qq import QQDisplay
from generalized_additive_models.inspection.residual_histogram import ResidualHistogramDisplay
from generalized_additive_models.inspection.residual_scatter import ResidualScatterDisplay

__all__ = [
    "ResidualHistogramDisplay",
    "QQDisplay",
    "ResidualScatterDisplay",
    "PartialEffectDisplay",
]
