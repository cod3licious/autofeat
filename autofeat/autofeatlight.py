# -*- coding: utf-8 -*-
# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler, PowerTransformer


def _check_features(df, corrthr=0.995, verbose=0):
    """
    Identify features with zeros variance or a correlation of (almost) 1 to other features, i.e., useless features.

    Inputs:
        - df: pd.DataFrame with all features
        - corrthr: threshold for correlations: if a feature has a higher pearson correlation to another feature it's
                   considered as redundant and ignored (float; default: 0.995)
        - verbose: verbosity level (int; default: 0)
    Returns:
        - list of column names representing 'ok' features (numeric, non-zero variance, not redundant)
    """
    # make sure all data is numeric
    df = df.select_dtypes(include=np.number)
    useless_cols = set()
    # 1. identify features with zero variance
    for c, v in df.var().items():
        if pd.isna(v) or v <= sys.float_info.epsilon:
            useless_cols.add(c)
    # 2. identify redundant features (that have a correlation of ~1 with other features)
    correlated_cols = defaultdict(set)
    corrmat = df.corr().abs()
    np.fill_diagonal(corrmat.values, 0)
    for c, v in corrmat.unstack().sort_values(ascending=False).items():
        if v < corrthr:
            break
        if (c[0] != c[1]) and (c[0] not in useless_cols):
            correlated_cols[c[0]].add(c[1])
    if correlated_cols:
        for c in df.columns:
            # the first variable that is correlated with others adds its correlated variables to the set of useless cols
            # since we check if a variable is in useless_cols, the correlated variables can't add the original variable
            if (c not in useless_cols) and (c in correlated_cols):
                useless_cols.update(correlated_cols[c])
    # return list of columns that should be kept
    if verbose:
        print("[AutoFeatLight] %i columns identified as useless:" % len(useless_cols))
        if useless_cols:
            print(sorted(useless_cols))
    return [c for c in df.columns if c not in useless_cols]


def _compute_additional_features(X, feature_names=None, compute_ratio=True, compute_product=True, verbose=0):
    """
    Compute additional non-linear features from the original features (ratio or product of two features).

    Inputs:
        - X: np.array with data (n_datapoints x n_features)
        - feature_names: optional list of column names to identify the features in X
        - compute_ratio: bool (default: True), whether to compute ratios of features
        - compute_product: bool (default: True), whether to compute products of features
        - verbose: verbosity level (int; default: 0)
    Returns:
        - np.array (n_datapoints x n_additional_features) with newly computed features
        - list with n_additional_features names describing the newly computed features
    """
    # check how many new features we will compute
    d = X.shape[1]
    n = 0
    if compute_ratio:
        n += d*d - d
    if compute_product:
        n += (d*d - d)//2
    if not n:
        print("ERROR: call _compute_additional_features with compute_ratio and/or compute_product set to True")
        return None, []
    if not feature_names:
        feature_names = ["x%i" % i for i in range(1, d+1)]
    # compute new features
    if verbose:
        print("[AutoFeatLight] computing %i additional features from %i original features" % (n, d))
    new_features = []
    X_new = np.zeros((X.shape[0], n))
    new_i = 0
    if compute_ratio:
        for i in range(d):
            # compute 1/x1
            with np.errstate(divide='ignore'):
                x = 1/X[:, i]
            # instead of dividing by 0 for some data points we just set the new feature to 0 there
            x[np.invert(np.isfinite(x))] = 0.
            for j in range(d):
                if i != j:
                    # multiply with x2 to get x2/x1
                    X_new[:, new_i] = x * X[:, j]
                    new_features.append("%s / %s" % (feature_names[j], feature_names[i]))
                    new_i += 1
    if compute_product:
        for i in range(d):
            for j in range(i+1, d):
                X_new[:, new_i] = X[:, i] * X[:, j]
                new_features.append("%s * %s" % (feature_names[i], feature_names[j]))
                new_i += 1
    assert new_i == n, "Internal Error in _compute_additional_features: new_i: %r, n: %r" % (new_i, n)
    return X_new, new_features


class AutoFeatLight(BaseEstimator):

    def __init__(
        self,
        compute_ratio=True,
        compute_product=True,
        scale=True,
        power_transform=True,
        corrthr_init=0.99999,
        corrthr=0.995,
        verbose=0,
    ):
        """
        Basic Feature Engineering:
            - remove useless features (zero variance or redundant)
            - compute additional non-linear features (ratios and product of original features, i.e. x1/x2 and x1*x2)
            - make all features more normally distributed (using the yeo-johnson power transform)

        Inputs:
            - compute_ratio: bool (default: True), whether to compute ratios of features
            - compute_product: bool (default: True), whether to compute products of features
            - scale: bool (default: True), rudimentary scaling of the data (only relevant if not computing the power_transform anyways)
            - power_transform: bool (default: True), whether to use a power transform (yeo-johnson) to make all features more normally distributed
            - corrthr_init: threshold for initial features (see below; float; default: 0.99999)
            - corrthr: threshold for correlations: if a feature has a higher pearson correlation to another feature it's
                       considered as redundant and ignored (float; default: 0.995)
            - verbose: verbosity level (int; default: 0)

        Attributes (after calling fit/fit_transform):
            - features_: feature names of transformed features
            - original_columns_: original columns of X when calling fit
            - return_df_: whether fit was called with a dataframe in which case a df will be returned as well,
                          otherwise a numpy array
            - good_cols_org_: list of good features from the original inputs
            - scaler_: if scale: fitted standard scaler
            - power_transformer_: if power_transform: fitted power transform
        """
        self.compute_ratio = compute_ratio
        self.compute_product = compute_product
        self.scale = scale
        self.power_transform = power_transform
        self.corrthr_init = corrthr_init
        self.corrthr = corrthr
        self.verbose = verbose

    def fit(self, X):
        """
        WARNING: call fit_transform instead!

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        """
        if self.verbose:
            print("[AutoFeatLight] Warning: This just calls fit_transform() but does not return the transformed dataframe.")
            print("[AutoFeatLight] It is much more efficient to call fit_transform() instead of fit() and transform()!")
        _ = self.fit_transform(X)  # noqa
        return self

    def transform(self, X):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        Returns:
            - new_X: new pandas dataframe or numpy array with additional/transformed features
        """
        check_is_fitted(self, ["good_cols_org_"])
        if not self.good_cols_org_:
            if self.verbose > 0:
                print("[AutoFeatLight] WARNING: No good features found; returning data unchanged.")
            return X
        if isinstance(X, pd.DataFrame):
            # make sure all data is numeric or we'll get an error when checking X
            X = X.select_dtypes(include=np.number)
            df_index = X.index
        else:
            df_index = None
        # check input
        cols = list(X.columns) if isinstance(X, pd.DataFrame) else ["x%i" % i for i in range(1, X.shape[1]+1)]
        X = check_array(X, force_all_finite="allow-nan")
        if not cols == self.original_columns_:
            raise ValueError("[AutoFeatLight] Not the same features as when calling fit.")
        # sort out useless original columns
        df = pd.DataFrame(X, columns=cols, index=df_index)[self.good_cols_org_]
        if self.compute_ratio or self.compute_product:
            # compute additional useful features
            X_new, new_features = _compute_additional_features(df.to_numpy(), self.good_cols_org_, self.compute_ratio, self.compute_product, self.verbose)
            df = pd.concat([df, pd.DataFrame(X_new, columns=new_features)], axis=1)
            df = df[self.features_]
        # scale/transform
        if self.scale or self.power_transform:
            X_new = self.scaler_.transform(df.to_numpy())
            if self.power_transform:
                X_new = self.power_transformer_.transform(X_new)
            df = pd.DataFrame(X_new, columns=df.columns, index=df.index)
        # return either dataframe or array
        return df if self.return_df_ else df.to_numpy()

    def fit_transform(self, X):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        Returns:
            - new_X: new pandas dataframe or numpy array with additional/transformed features
        """
        self.return_df_ = isinstance(X, pd.DataFrame)
        if isinstance(X, pd.DataFrame):
            # make sure all data is numeric or we'll get an error when checking X
            X = X.select_dtypes(include=np.number)
            df_index = X.index
        else:
            df_index = None
        # store column names as they'll be lost in the other check
        self.original_columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else ["x%i" % i for i in range(1, X.shape[1]+1)]
        # check input
        X = check_array(X, force_all_finite="allow-nan")
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=self.original_columns_, index=df_index)
        # see which of the original features are not completely useless
        self.good_cols_org_ = _check_features(df, self.corrthr_init, self.verbose)
        if not self.good_cols_org_:
            if self.verbose > 0:
                print("[AutoFeatLight] WARNING: No good features found; returning original features.")
            return df if self.return_df_ else X
        # compute additional features
        df = df[self.good_cols_org_]
        if self.compute_ratio or self.compute_product:
            X_new, new_features = _compute_additional_features(df.to_numpy(), self.good_cols_org_, self.compute_ratio, self.compute_product, self.verbose)
            # add new features to original dataframe
            df = pd.concat([df, pd.DataFrame(X_new, columns=new_features, index=df_index)], axis=1)
            # check again which of the features we should keep
            self.features_ = _check_features(df, self.corrthr, self.verbose)
            df = df[self.features_]
        else:
            self.features_ = self.good_cols_org_
        if self.scale or self.power_transform:
            # scale data to avoid errors due to large numbers
            self.scaler_ = StandardScaler(with_mean=False)
            X_new = self.scaler_.fit_transform(df.to_numpy())
            if self.power_transform:
                self.power_transformer_ = PowerTransformer(method='yeo-johnson', standardize=True)
                X_new = self.power_transformer_.fit_transform(X_new)
            df = pd.DataFrame(X_new, columns=df.columns, index=df.index)
        # return either dataframe or array
        return df if self.return_df_ else df.to_numpy()
