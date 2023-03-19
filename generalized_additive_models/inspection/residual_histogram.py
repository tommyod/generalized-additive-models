#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


class ResidualHistogramDisplay:
    def __init__(self, *, residuals):
        self.residuals = residuals

    def plot(
        self,
        ax=None,
        *,
        bin_kwargs=None,
    ):
        if bin_kwargs is None:
            bin_kwargs = {}

        default_bin_kwargs = {"bins": "auto", "zorder": 10}
        bin_kwargs = {**default_bin_kwargs, **bin_kwargs}

        if ax is None:
            _, ax = plt.subplots()

        self.n_, self.bins_, self.patches_ = ax.hist(self.residuals, **bin_kwargs)

        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    def from_estimator(
        cls,
        gam,
        X,
        y,
        *,
        residuals="deviance",
        standardized=True,
        ax=None,
        bin_kwargs=None,
    ):
        residuals = gam.residuals(
            X,
            y,
            residuals=residuals,
            standardized=standardized,
        )

        viz = cls(
            residuals=residuals,
        )

        return viz.plot(
            ax=ax,
            bin_kwargs=bin_kwargs,
        )
