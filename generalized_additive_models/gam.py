#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:05:41 2023

@author: tommy
"""

import copy
import functools
import sys
import warnings
from numbers import Integral, Real

import numpy as np
import tabulate
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_consistent_length, check_scalar, column_or_1d
from sklearn.utils._param_validation import Hidden, Interval, StrOptions

# https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/utils/validation.py#L1870
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from generalized_additive_models.distributions import DISTRIBUTIONS, Distribution
from generalized_additive_models.links import LINKS, Link
from generalized_additive_models.optimizers import LBFGSB, PIRLS
from generalized_additive_models.terms import Categorical, Intercept, Linear, Spline, Tensor, Term, TermList


class GAM(BaseEstimator):
    """Initialize a Generalized Additive Model (GAM).

    Parameters
    ----------
    terms : Term, TermList or list, optional
        The term(s) of the model. The argument can be a single term or a
        collection of terms. The features that the terms refer to must be
        present in the data set at fit and predict time.
        The default is None.
    distribution : str or Distribution, optional
        The assumed distribution of the target variable. Look at the dict
        GAM.DISTRIBUTIONS for a list of available options.
        The default is "normal".
    link : str or Link, optional
        The assumed link function of the target variable. Look at the dict
        GAM.LINKS for a list of available options.
        The default is "identity".
    fit_intercept : bool, optional
        Whether or not to automatically add an intercept term to the terms.
        If an intercept is already present, then this setting has no effect.
        The default is True.
    solver : str, optional
        Either 'pirls' or 'lbfgsb'.
        'pirls' stands for Penalized Iterated Reweighted Least Squares, which
        is a Newton routine that uses step-halving line search.
        'lbfgsb' is the Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm
        from scipy.optimize.minimize.
        The default is "pirls".
    max_iter : int, optional
        Maximum number of iterations in the solver.
        The default is 100.
    tol : TYPE, optional
        Tolerance in the solver.
        The default is 0.0001.
    verbose : int, optional
        Verbosity level. The higher the number, the more info is printed.
        The default is 0.

    Returns
    -------
    None.

    Examples
    --------
    >>> from generalized_additive_models import GAM, Spline, Categorical
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes(as_frame=True)
    >>> df = data.data
    >>> y = data.target
    >>> gam = GAM(Spline("age") + Spline("bmi") + Spline("bp") + Categorical("sex"))
    >>> gam = gam.fit(df, y)
    >>> predictions = gam.predict(df)
    >>> for term in gam.terms:
    ...     print(term, term.coef_) # doctest: +SKIP

    """

    DISTRIBUTIONS = DISTRIBUTIONS
    LINKS = LINKS

    _parameter_constraints: dict = {
        "terms": [Term, TermList],
        "distribution": [StrOptions(set(DISTRIBUTIONS.keys())), Distribution],
        "link": [StrOptions(set(LINKS.keys())), Link],
        "fit_intercept": ["boolean"],
        "solver": [
            StrOptions({"pirls", "lbfgsb"}),
            Hidden(type),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "verbose": [Integral, "boolean"],
    }

    def __init__(
        self,
        terms=None,
        *,
        distribution="normal",
        link="identity",
        fit_intercept=True,
        solver="pirls",
        max_iter=100,
        tol=0.0001,
        verbose=0,
    ):
        self.terms = terms
        self.distribution = distribution
        self.link = link
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _validate_params(self, X):
        super()._validate_params()

        # if not hasattr(self, "_distribution"):
        self._link = LINKS[self.link]() if isinstance(self.link, str) else self.link

        # if not hasattr(self, "_distribution"):
        self._distribution = (
            DISTRIBUTIONS[self.distribution]() if isinstance(self.distribution, str) else self.distribution
        )

        if self.solver == "pirls":
            self._solver = PIRLS
        elif self.solver == "lbfgsb":
            self._solver = LBFGSB
        else:
            raise ValueError("Unknown solver.")

        self.terms = TermList(self.terms)

        # Auto model
        # If only a single Term is passed, and that term has `feature=None`,
        # then expand and use one term per column with the other parameters
        num_samples, num_features = X.shape
        if len(self.terms) == 1:
            term = self.terms[0]
            if isinstance(term, Spline) and term.feature is None:
                term_params = term.get_params()
                term_params.pop("feature")
                self.terms = TermList([Spline(feature=i, **term_params) for i in range(num_features)])
            elif isinstance(term, Linear) and term.feature is None:
                term_params = term.get_params()
                term_params.pop("feature")
                self.terms = TermList([Linear(feature=i, **term_params) for i in range(num_features)])

        if self.fit_intercept and (Intercept() not in self.terms):
            self.terms.append(Intercept())

    def _get_sample_weight(self, *, y=None, mu=None, sample_weight=None):
        """Return sample weight. The arguments `y` and `mu` are used for
        compatibility with ExpectileGAM."""
        return sample_weight

    def fit(self, X, y, sample_weight=None):
        """Fit model to data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset to fit to. Must be a np.ndarray of dimension 2 with shape
            (num_samples, num_features) or a pandas DataFrame. If the `terms`
            in the GAM refer to integer features, a np.ndarray must be passed.
            If the `terms` refer to string column names, a pandas DataFrame must
            be passed.
        y : np.ndarray or Series
            An array of target values.
        sample_weight : np.ndarray, optional
            An array of sample weights. Sample weights [1, 3] is equal to
            repeating the second data point three times.
            The default is None.

        Returns
        -------
        GAM
            Returns the instance.

        Examples
        --------
        >>> rng = np.random.default_rng(32)
        >>> X = rng.normal(size=(100, 1))
        >>> y = np.sin(X).ravel()
        >>> gam = GAM(Spline(0))
        >>> gam.fit(X, y)
        GAM(terms=TermList(data=[Spline(feature=0), Intercept()]))

        """
        self._validate_params(X)
        check_consistent_length(X, y, sample_weight)
        y = column_or_1d(y)

        sample_weight = _check_sample_weight(sample_weight, X, only_non_negative=True)
        sample_weight = column_or_1d(sample_weight)

        self.model_matrix_ = self.terms.fit_transform(X)

        self.X_ = X.copy()  # Store a copy used for partial effects
        self.y_ = y.copy()
        self.sample_weight_ = sample_weight.copy()

        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=float)

        optimizer = self._solver(
            X=self.model_matrix_,
            D=self.terms.penalty_matrix(),
            y=y,
            link=self._link,
            distribution=self._distribution,
            bounds=(self.terms._lower_bound, self.terms._upper_bound),
            get_sample_weight=functools.partial(self._get_sample_weight, sample_weight=sample_weight),
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )

        # Copy over solver information
        self.coef_ = optimizer.solve().copy()
        self.results_ = copy.deepcopy(optimizer.results_)
        self.results_.pseudo_r2 = self.score(X, y, sample_weight=sample_weight)

        # Update distribution scale if set to None
        self._distribution.scale = self.results_.scale if self._distribution.scale is None else self._distribution.scale
        assert self._distribution.scale is not None

        # Assign coefficients to terms
        coef_idx = 0
        for term in self.terms:
            term.coef_ = self.coef_[coef_idx : coef_idx + term.num_coefficients]
            term.coef_idx_ = np.arange(coef_idx, coef_idx + term.num_coefficients)
            term.coef_covar_ = self.results_.covariance[np.ix_(term.coef_idx_, term.coef_idx_)]
            term.edof_ = self.results_.edof_per_coef[term.coef_idx_]

            coef_idx += term.num_coefficients
            assert len(term.coef_) == term.num_coefficients
        assert sum(len(term.coef_) for term in self.terms) == len(self.coef_)

        return self

    def sample(self, mu, size=None, random_state=None):
        """Sample from the posterior predictive distribution.

        Parameters
        ----------
        mu : np.ndarray
            The expected value of the distribution.
        size : int, optional
            Number of samples. The default is None, which means one sample.
        random_state : int, optional
            Random state used to sample from the scipy distribution with
            reproducible results.

        Returns
        -------
        np.ndarray
            A numpy array of samples.

        Examples
        --------
        >>> rng = np.random.default_rng(32)
        >>> X = np.ones((9999, 1))
        >>> y = rng.normal(size=9999) * 10
        >>> gam = GAM(terms=Intercept(), fit_intercept=False).fit(X, y)
        >>> gam._distribution.scale
        99.389...
        >>> gam.sample(mu=np.zeros(5), random_state=1).round(1)
        array([ 16.2,  -6.1,  -5.3, -10.7,   8.6])
        >>> gam.sample(mu=np.zeros(5), random_state=3, size=(3, 5)).round(1)
        array([[ 17.8,   4.4,   1. , -18.6,  -2.8],
               [ -3.5,  -0.8,  -6.3,  -0.4,  -4.8],
               [-13.1,   8.8,   8.8,  17. ,   0.5]])

        """
        check_is_fitted(self, attributes=["coef_"])
        distr = self._distribution.to_scipy(mu)
        return distr.rvs(size=size, random_state=random_state)

    def predict(self, X):
        r"""Predict the expected value :math:`\mu` with the model.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset to predict on. Must be a np.ndarray of dimension 2 with
            shape (num_samples, num_features) or a pandas DataFrame. If the
            `terms` in the GAM refer to integer features, a np.ndarray must be
            passed. If the `terms` refer to string column names, a pandas
            DataFrame must be passed.

        Returns
        -------
        np.ndarray
            An array with predictions. Predictions are inverse_link(X @ coef).

        Examples
        --------

        Create a data set where probability of y being 1 increases with X:

        >>> rng = np.random.default_rng(32)
        >>> X = rng.normal(size=(99, 1))
        >>> probs = (1 / (1 +  np.exp(-X))).ravel()
        >>> y = rng.binomial(1, p=probs)

        Fit a model and predict:

        >>> gam = GAM(Linear(0), distribution="binomial", link="logit").fit(X, y)
        >>> X_new = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1)
        >>> gam.predict(X_new).round(3)
        array([0.011, 0.045, 0.168, 0.465, 0.789, 0.942, 0.986])

        """
        check_is_fitted(self, attributes=["coef_"])

        model_matrix = self.terms.transform(X)
        return self._link.inverse_link(model_matrix @ self.coef_)

    def score(self, X, y, sample_weight=None):
        """Proportion deviance explained (pseudo :math:`r^2`).


        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            A dataset to predict on. Must be a np.ndarray of dimension 2 with
            shape (num_samples, num_features) or a pandas DataFrame. If the
            `terms` in the GAM refer to integer features, a np.ndarray must be
            passed. If the `terms` refer to string column names, a pandas
            DataFrame must be passed.
        y : np.ndarray or Series
            An array of target values.
        sample_weight : np.ndarray, optional
            An array of sample weights. Sample weights [1, 3] is equal to
            repeating the second data point three times.
            The default is None.

        Returns
        -------
        float
            The pseudo r^2 score.

        """
        check_is_fitted(self, attributes=["coef_"])

        check_consistent_length(X, y, sample_weight)
        y = column_or_1d(y)

        sample_weight = _check_sample_weight(sample_weight, X, only_non_negative=True)
        sample_weight = column_or_1d(sample_weight)

        # Special case for the null gam. Without this line of code, a cycle of
        # fit() -> score() -> fit() will occur.
        if self.terms == TermList([Intercept()]):
            return 0

        # Compute pseudo r2
        # https://en.wikipedia.org/wiki/Pseudo-R-squared#R2L_by_Cohen
        # Page 128 in Wood, 2nd edition
        mu = self.predict(X)
        sample_weight = self._get_sample_weight(y=y, mu=mu, sample_weight=sample_weight)

        # Compute the null predictions
        null_gam = clone(self)
        null_preds = null_gam.set_params(terms=Intercept(), verbose=0).fit(X, y, sample_weight=sample_weight).predict(X)

        null_deviance = self._distribution.deviance(y=y, mu=null_preds, sample_weight=sample_weight).sum()
        fitted_deviance = self._distribution.deviance(y=y, mu=mu, sample_weight=sample_weight).sum()
        # assert fitted_deviance <= null_deviance
        return (null_deviance - fitted_deviance) / null_deviance

    def summary(self, file=None):
        """Print a model summary.

        Parameters
        ----------
        file : filehandle, optional
            A file handle to write to.
            The default is None, which maps to sys.stdout.

        Returns
        -------
        None.

        """
        check_is_fitted(self, attributes=["coef_"])

        if file is None:
            file = sys.stdout

        p = functools.partial(print, file=file)
        fmt = functools.partial(np.format_float_positional, precision=3, min_digits=3)

        # ======================= GAM PROPERTIES =======================
        rows = []
        rows.append(("Model", type(self).__name__))
        rows.append(("Link", self._link))
        rows.append(("Distribution", type(self._distribution).__name__))
        rows.append(("Scale", fmt(self.results_.scale)))
        rows.append(("GCV", fmt(self.results_.generalized_cross_validation_score)))
        rows.append(("Explained deviance", fmt(self.results_.pseudo_r2)))

        gam_table_str = tabulate.tabulate(
            rows,
            headers=("Property", "Value"),
            tablefmt="github",
        )

        p(gam_table_str)

        # ======================= TERM PROPERTIES =======================

        rows = []
        for term in self.terms:
            t_name = type(term).__name__
            t_features = ", ".join(s.feature for s in term) if isinstance(term, Tensor) else (term.feature or "")
            t_repr = f"{t_name}({t_features})"
            t_numcoef = term.num_coefficients
            t_edof = fmt(term.edof_.sum())
            rows.append((t_repr, t_numcoef, t_edof))

        p()
        term_table_str = tabulate.tabulate(
            rows,
            headers=("Term", "Coefs", "Edof"),
            tablefmt="github",
        )
        p(term_table_str)

        # ============================ COEFFICIENTS ============================
        fmt = functools.partial(np.format_float_positional, precision=4, min_digits=4)

        rows = []
        for term in self.terms:
            if isinstance(term, Intercept):
                t_name = type(term).__name__
                t_repr = f"{t_name}()"
                t_coef = term.coef_[0]
                t_std = np.sqrt(term.coef_covar_[0])
                rows.append((t_repr, fmt(t_coef), fmt(t_std)))

            elif isinstance(term, Linear):
                t_name = type(term).__name__
                t_feature = term.feature
                t_repr = f"{t_name}({t_feature})"
                t_coef = term.coef_[0]
                t_std = np.sqrt(term.coef_covar_[0])
                rows.append((t_repr, fmt(t_coef), fmt(t_std)))

            elif isinstance(term, Categorical):
                t_name = type(term).__name__
                t_feature = term.feature

                for i, category in enumerate(term.categories_):
                    t_repr = f"{t_name}({t_feature}={category})"
                    t_coef = term.coef_[i]
                    t_std = np.sqrt(term.coef_covar_[i, i])
                    rows.append((t_repr, fmt(t_coef), fmt(t_std)))
            else:
                continue

        p()
        coef_table_str = tabulate.tabulate(
            rows,
            headers=("Term", "Coef", "Coef std"),
            tablefmt="github",
        )
        p(coef_table_str)

    def residuals(self, X, y, *, residuals="deviance", standardized=True):
        check_is_fitted(self, attributes=["coef_"])

        distribution = self._distribution

        y_pred = self.predict(X)
        y_true = np.array(y, dtype=float)

        expected_residuals = {"response", "pearson", "deviance"}

        if residuals == "response":
            residuals = y_true - y_pred
            if standardized:
                residuals = residuals / np.sqrt(distribution.variance(y_pred))

        # Se page 112 in Wood, 2nd edition
        elif residuals == "pearson":
            # Should have approximately zero mean and variance "scale"
            residuals = (y_true - y_pred) / np.sqrt(distribution.V(y_pred))
            # Should be N(0, 1)
            if standardized:
                residuals = residuals / np.sqrt(distribution.scale)

        elif residuals == "deviance":
            deviances = distribution.deviance(y=y_true, mu=y_pred, scaled=False)
            residuals = np.sqrt(deviances) * np.sign(y_true - y_pred)
            # Should be N(0, 1)
            if standardized:
                residuals = residuals / np.sqrt(distribution.scale)
        else:
            raise ValueError(f"Parameter `residuals` must be in: {expected_residuals}")

        return residuals


class ExpectileGAM(GAM):
    r"""Initialize an ExpectileGAM.
    
    A GAM with a Normal distribution and an Identity link minimizes a weighted
    least squares objective
    
    .. math::
        
       \ell(\beta) = \sum_i w_i (f(x_i; \beta) - y_i)^2 + \operatorname{penalty}(\beta)
       
    The ExpectileGAM minimizes a least asymmetrically weighted squares objective.
    The weights :math:`w_i` are chosen based on the residuals
    :math:`\epsilon_i = f(x_i; \beta) - y_i` and a desired `expectile` :math:`\tau`. 
    The weights are given by
    
    .. math::
        
        \epsilon_i = \begin{cases}
          \tau  &\text{ if } \epsilon_i \leq 0 \\
          1 - \tau &\text{ if } \epsilon_i > 0
        \end{cases}
        
    For more information, see:
        
        - https://freakonometrics.hypotheses.org/files/2017/05/erasmus-1.pdf
        
    Parameters
    ----------
    terms : Term, TermList or list, optional
        The term(s) of the model. The argument can be a single term or a
        collection of terms. The features that the terms refer to must be
        present in the data set at fit and predict time. 
        The default is None.
    expectile : float, optional
        The expectile to fit to.
        The default is 0.5.
    fit_intercept : TYPE, optional
        Whether or not to automatically add an intercept term to the terms.
        If an intercept is already present, then this setting has no effect.
        The default is True.
    solver : str, optional
        The solver to use, currently only "pirls" is available, which stands
        for Penalized Iterated Reweighted Least Squares. 
        The default is "pirls".
    max_iter : int, optional
        Maximum number of iterations in the solver. 
        The default is 100.
    tol : TYPE, optional
        Tolerance in the solver. 
        The default is 0.0001.
    verbose : int, optional
        Verbosity level. The higher the number, the more info is printed. 
        The default is 0.

    Returns
    -------
    None.
    """

    _parameter_constraints: dict = {
        "terms": [Term, TermList],
        "expectile": [Interval(Real, 0, 1, closed="neither")],
        "distribution": [StrOptions(set(DISTRIBUTIONS.keys())), Distribution],
        "link": [StrOptions(set(LINKS.keys())), Link],
        "fit_intercept": ["boolean"],
        "solver": [
            StrOptions({"pirls"}),
            Hidden(type),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "verbose": [Integral, "boolean"],
    }

    def __init__(
        self,
        terms=None,
        *,
        expectile=0.5,
        fit_intercept=True,
        solver="pirls",
        max_iter=100,
        tol=0.0001,
        verbose=-1,
    ):
        self.expectile = expectile
        super().__init__(
            terms=terms,
            distribution="normal",
            link="identity",
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )

    def _validate_params(self, X):
        super()._validate_params(X)

    def _get_sample_weight(self, *, y=None, mu=None, sample_weight=None):
        if (y is None) and (mu is None):
            return sample_weight

        # asymmetric weight
        # see slide 8
        # https://freakonometrics.hypotheses.org/files/2017/05/erasmus-1.pdf
        asymmetric_weights = (y > mu) * self.expectile + (y <= mu) * (1 - self.expectile)

        return sample_weight * asymmetric_weights

    def fit_quantile(self, X, y, quantile, max_iter=20, tol=0.01, sample_weight=None):
        """Find the `expectile` such that the empirical quantile matches `quantile`.

        Finding the desired `expectile` is done by performing binary search.

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors, where n_samples is the number of samples
            and m_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.
        quantile : float on (0, 1)
            desired quantile to fit.
        max_iter : int, default: 20
            maximum number of binary search iterations to perform
        tol : float > 0, default: 0.01
            maximum distance between desired quantile and fitted quantile
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        self : Fitted GAM object with updated `expectile` parameter.

        Examples
        --------
        >>> import numpy as np
        >>> from generalized_additive_models import ExpectileGAM, Intercept
        >>> rng = np.random.default_rng(42)
        >>> X = np.ones((1000, 1))
        >>> y = rng.triangular(left=0, mode=0.5, right=1, size=(1000, 1))
        >>> gam = ExpectileGAM(Intercept()).fit_quantile(X, y, quantile=0.9)
        >>> (gam.predict(X) > y).mean()
        0.90...
        >>> gam.expectile
        0.976...

        """

        quantile = check_scalar(
            quantile,
            "quantile",
            Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )

        max_iter = check_scalar(
            max_iter,
            "quantile",
            Integral,
            min_val=1,
            include_boundaries="left",
        )

        tol = check_scalar(
            tol,
            "quantile",
            Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )

        def _within_tol(a, b, tol):
            return abs(a - b) <= tol

        # Perform binary search
        # The goal is to choose `expectile` such that the empirical quantile
        # matches the desired quantile. The reason for not using
        # scipy.optimize.bisect is that bisect evaluates the endpoints first,
        # resulting in extra unnecessary fits (we assume that 0 -> 0 and 1 -> 1)
        min_, max_ = 0.0, 1.0
        expectile = self.expectile
        # TODO: Can this be improved?
        # https://en.wikipedia.org/wiki/ITP_method
        for iteration in range(max_iter):
            # Fit the model and compute the expected quantile
            self.set_params(expectile=expectile)
            self.fit(X, y, sample_weight=sample_weight)
            empirical_quantile = (self.predict(X) > y).mean()

            # Print out information
            if self.verbose >= 0:
                digits = 4
                expectile_fmt = np.format_float_positional(
                    self.expectile,
                    precision=digits,
                    pad_right=digits,
                    min_digits=digits,
                )
                empir_quant_fmt = np.format_float_positional(
                    empirical_quantile,
                    precision=digits,
                    pad_right=digits,
                    min_digits=digits,
                )
                quantile_fmt = np.format_float_positional(
                    quantile, precision=digits, pad_right=digits, min_digits=digits
                )
                msg = f"{iteration}  Fitting with expectile={expectile_fmt} gave empirical "
                msg += f"quantile {empir_quant_fmt} (target={quantile_fmt})."
                print(msg)

            if _within_tol(empirical_quantile, quantile, tol):
                break

            if empirical_quantile < quantile:
                min_ = self.expectile  # Move up
            else:
                max_ = self.expectile  # Move down

            expectile = (min_ + max_) / 2.0

        # print diagnostics
        if not _within_tol(empirical_quantile, quantile, tol):
            msg = f"Could determine `expectile` within tolerance {tol} in {max_iter} iterations.\n"
            msg += f"Ended up with `expectile={expectile}`, which gives an empirical\n"
            msg += f"quantile of {empirical_quantile} (desired quantile was {quantile})."
            warnings.warn(msg, ConvergenceWarning)

        return self


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "--maxfail=1"])
