#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


class ResidualScatterDisplay:
    """Plot residuals."""

    def __init__(self, *, x, residuals):
        self.x = x
        self.residuals = residuals

    def plot(
        self,
        ax=None,
        *,
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        if scatter_kwargs is None:
            scatter_kwargs = {}
        if line_kwargs is None:
            line_kwargs = {}

        default_scatter_kwargs = {"alpha": 0.8}
        default_line_kwargs = {"color": "black", "alpha": 0.7, "linestyle": "--"}

        scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}
        line_kwargs = {**default_line_kwargs, **line_kwargs}

        if ax is None:
            _, ax = plt.subplots()

        self.scatter_ = ax.scatter(self.x, self.residuals, **scatter_kwargs)

        self.line_ = ax.plot(
            [np.min(self.x), np.max(self.x)],
            [0, 0],
            **line_kwargs,
        )[0]

        ax.set(ylabel="Residuals")

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
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        residuals = gam.residuals(X, y, residuals=residuals, standardized=standardized)
        y_pred = gam.predict(X)

        viz = cls(
            x=y_pred,
            residuals=residuals,
        )

        return viz.plot(
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            line_kwargs=line_kwargs,
        )


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    from generalized_additive_models import GAM, Categorical, Spline

    data = load_diabetes(as_frame=True)
    df = data.data
    y = data.target
    gam = GAM(Spline("age") + Spline("bmi") + Spline("bp") + Categorical("sex"))
    gam = gam.fit(df, y)
    residuals = gam.residuals(df, y)

    ResidualScatterDisplay(x=df["age"], residuals=residuals).plot()
    ResidualScatterDisplay(x=df["bmi"], residuals=residuals).plot()
