#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numbers

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import _safe_indexing, check_random_state


class ResidualScatterDisplay:
    def __init__(self, *, residuals, y_pred):
        self.residuals = residuals
        self.y_pred = y_pred

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

        self.scatter_ = ax.scatter(self.y_pred, self.residuals, **scatter_kwargs)

        self.line_ = ax.plot(
            [np.min(self.y_pred), np.max(self.y_pred)],
            [0, 0],
            **line_kwargs,
        )[0]

        xlabel, ylabel = "Predicted values", "Residuals"
        ax.set(xlabel=xlabel, ylabel=ylabel)

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
        subsample=1_000,
        random_state=None,
        ax=None,
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        random_state = check_random_state(random_state)

        n_samples = len(y)
        if isinstance(subsample, numbers.Integral):
            if subsample <= 0:
                raise ValueError(f"When an integer, subsample={subsample} should be positive.")
        elif isinstance(subsample, numbers.Real):
            if subsample <= 0 or subsample >= 1:
                raise ValueError(f"When a floating-point, subsample={subsample} should" " be in the (0, 1) range.")
            subsample = int(n_samples * subsample)
        elif subsample is None:
            y_sampled = y
            X_sampled = X
            subsample = n_samples

        if subsample is not None and subsample < n_samples:
            indices = random_state.choice(np.arange(n_samples), size=subsample)
            y_sampled = _safe_indexing(y, indices, axis=0)
            X_sampled = _safe_indexing(X, indices, axis=0)

        residuals = gam.residuals(X_sampled, y_sampled, residuals=residuals, standardized=standardized)
        y_pred = gam.predict(X_sampled)

        viz = cls(
            residuals=residuals,
            y_pred=y_pred,
        )

        return viz.plot(
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            line_kwargs=line_kwargs,
        )
