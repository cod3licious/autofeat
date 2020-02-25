# -*- coding: utf-8 -*-
# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import zip
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import sklearn.linear_model as lm


def _add_noise_features(X):
    """
    Adds 3-1.5*d additional noise features to X.

    Inputs:
        - X: n x d numpy array with d features
    Returns:
        - X with additional noise features
    """
    n_feat = X.shape[1]
    if X.shape[0] > 50 and n_feat > 1:
        # shuffled features
        rand_noise = StandardScaler().fit_transform(np.random.permutation(X.flatten()).reshape(X.shape))
        X = np.hstack([X, rand_noise])
    # normally distributed noise
    rand_noise = np.random.randn(X.shape[0], max(3, int(0.5*n_feat)))
    X = np.hstack([X, rand_noise])
    return X


def _noise_filtering(X, target, good_cols=[], problem_type="regression"):
    """
    Trains a prediction model with additional noise features and selects only those of the
    original features that have a higher coefficient than any of the noise features.

    Inputs:
        - X: n x d numpy array with d features
        - target: n dimensional array with targets corresponding to the data points in X
        - good_cols: list of column names for the features in X
        - problem_type: str, either "regression" or "classification" (default: "regression")
    Returns:
        - good_cols: list of noise filtered column names
    """
    n_feat = X.shape[1]
    assert len(good_cols) == n_feat, "fewer column names provided than features in X."
    if not good_cols:
        good_cols = list(range(n_feat))
    # perform noise filtering on these features
    if problem_type == "regression":
        model = lm.LassoLarsCV(cv=5, eps=1e-8)
    elif problem_type == "classification":
        model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
    else:
        print("[featsel] WARNING: Unknown problem_type %r - not performing noise filtering." % problem_type)
        model = None
    if model is not None:
        X = _add_noise_features(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # TODO: remove if sklearn least_angle issue is fixed
            try:
                model.fit(X, target)
            except ValueError:
                rand_idx = np.random.permutation(X.shape[0])
                model.fit(X[rand_idx], target[rand_idx])
            # model.fit(X, target)
        if problem_type == "regression":
            coefs = np.abs(model.coef_)
        else:
            # model.coefs_ is n_classes x n_features, but we need n_features
            coefs = np.max(np.abs(model.coef_), axis=0)
        weights = dict(zip(good_cols, coefs[:len(good_cols)]))
        # only include features that are more important than our known noise features
        noise_w_thr = np.max(coefs[n_feat:])
        good_cols = [c for c in good_cols if weights[c] > noise_w_thr]
    return good_cols


def _select_features_1run(df, target, problem_type="regression", verbose=0):
    """
    One feature selection run.

    Inputs:
        - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
              to the n training data points. The variables have to be scaled to have 0 mean and unit variance.
        - target: n dimensional array with targets corresponding to the data points in df
        - problem_type: str, either "regression" or "classification" (default: "regression")
        - verbose: verbosity level (int; default: 0)
    Returns:
        - good_cols: list of column names for df with which a prediction model can be trained
    """
    # initial selection of too few but (hopefully) relevant features
    if problem_type == "regression":
        model = lm.LassoLarsCV(cv=5, eps=1e-8)
    elif problem_type == "classification":
        model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
    else:
        print("[featsel] WARNING: Unknown problem_type %r - not performing feature selection!" % problem_type)
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: remove if sklearn least_angle issue is fixed
        try:
            model.fit(df, target)
        except ValueError:
            # try once more with shuffled data, if it still doesn't work, give up
            rand_idx = np.random.permutation(df.shape[0])
            model.fit(df.iloc[rand_idx], target[rand_idx])
        # model.fit(df, target)
    if problem_type == "regression":
        coefs = np.abs(model.coef_)
    else:
        # model.coefs_ is n_classes x n_features, but we need n_features
        coefs = np.max(np.abs(model.coef_), axis=0)
    # weight threshold: select at most 0.2*n_train initial features
    thr = sorted(coefs, reverse=True)[min(df.shape[1]-1, df.shape[0]//5)]
    initial_cols = list(df.columns[coefs > thr])
    # noise filter
    initial_cols = _noise_filtering(df[initial_cols].to_numpy(), target, initial_cols, problem_type)
    good_cols = set(initial_cols)
    if verbose > 0:
        print("[featsel]\t %i initial features." % len(initial_cols))
    # add noise features
    X_w_noise = _add_noise_features(df[initial_cols].to_numpy())
    # go through all remaining features in splits of n_feat <= 0.5*n_train
    other_cols = list(np.random.permutation(list(set(df.columns).difference(initial_cols))))
    if other_cols:
        n_splits = int(np.ceil(len(other_cols)/max(10, 0.5*df.shape[0]-len(initial_cols))))
        split_size = int(np.ceil(len(other_cols)/n_splits))
        for i in range(n_splits):
            current_cols = other_cols[i*split_size:min(len(other_cols), (i+1)*split_size)]
            X = np.hstack([df[current_cols].to_numpy(), X_w_noise])
            if problem_type == "regression":
                model = lm.LassoLarsCV(cv=5, eps=1e-8)
            else:
                model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # TODO: remove if sklearn least_angle issue is fixed
                try:
                    model.fit(X, target)
                except ValueError:
                    rand_idx = np.random.permutation(X.shape[0])
                    model.fit(X[rand_idx], target[rand_idx])
                # model.fit(X, target)
            current_cols.extend(initial_cols)
            if problem_type == "regression":
                coefs = np.abs(model.coef_)
            else:
                # model.coefs_ is n_classes x n_features, but we need n_features
                coefs = np.max(np.abs(model.coef_), axis=0)
            weights = dict(zip(current_cols, coefs[:len(current_cols)]))
            # only include features that are more important than our known noise features
            noise_w_thr = np.max(coefs[len(current_cols):])
            good_cols.update([c for c in weights if abs(weights[c]) > noise_w_thr])
            if verbose > 0:
                print("[featsel]\t Split %2i/%i: %3i candidate features identified." % (i+1, n_splits, len(good_cols)), end="\r")
    # noise filtering on the combination of features
    good_cols = list(good_cols)
    good_cols = _noise_filtering(df[good_cols].to_numpy(), target, good_cols, problem_type)
    if verbose > 0:
        print("\n[featsel]\t Selected %3i features after noise filtering." % len(good_cols))
    return good_cols


def select_features(df, target, featsel_runs=5, keep=None, problem_type="regression", n_jobs=1, verbose=0):
    """
    Selects predictive features given the data and targets.

    Inputs:
        - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
              to the n training data points.
        - target: n dimensional array with targets corresponding to the data points in df
        - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
        - keep: list of features that should be kept no matter what
        - problem_type: str, either "regression" or "classification" (default: "regression")
        - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
        - verbose: verbosity level (int; default: 0)
    Returns:
        - good_cols: list of column names for df with which a regression model can be trained
    """
    if not (len(df) == len(target)):
        raise ValueError("[featsel] df and target dimension mismatch.")
    if keep is None:
        keep = []
    # check that keep columns are actually in df (- the columns might have been transformed to strings!)
    keep = [c for c in keep if c in df.columns and not str(c) in df.columns] + [str(c) for c in keep if str(c) in df.columns]
    # scale features to have 0 mean and unit std
    if verbose > 0:
        if featsel_runs > df.shape[0]:
            print("[featsel] WARNING: Less data points than featsel runs!!")
        print("[featsel] Scaling data...", end="")
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, dtype=np.float32)
        if problem_type == "regression":
            target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()
        else:
            target_scaled = target
    if verbose > 0:
        print("done.")

    good_cols = list(df.columns)

    # select good features in k runs in parallel
    # by doing sort of a cross-validation (i.e., randomly subsample data points)
    def run_select_features(i):
        if verbose > 0:
            print("[featsel] Feature selection run %i/%i" % (i+1, featsel_runs))
        np.random.seed(i)
        rand_idx = np.random.permutation(df_scaled.index)[:max(10, int(0.85 * len(df_scaled)))]
        return _select_features_1run(df_scaled.iloc[rand_idx], target_scaled[rand_idx], problem_type, verbose=verbose-1)

    if featsel_runs >= 1 and problem_type in ("regression", "classification"):
        if n_jobs == 1 or featsel_runs == 1:
            # only use parallelization code if you actually parallelize
            selected_columns = []
            for i in range(featsel_runs):
                selected_columns.extend(run_select_features(i))
        else:
            def flatten_lists(l):
                return [item for sublist in l for item in sublist]

            selected_columns = flatten_lists(Parallel(n_jobs=n_jobs, verbose=100*verbose)(delayed(run_select_features)(i) for i in range(featsel_runs)))

        if selected_columns:
            selected_columns = Counter(selected_columns)
            # sort by frequency, but down weight longer formulas to break ties
            selected_columns = sorted(selected_columns, key=lambda x: selected_columns[x] - 0.000001*len(str(x)), reverse=True)
            if verbose > 0:
                print("[featsel] %i features after %i feature selection runs" % (len(selected_columns), featsel_runs))
            # correlation filtering
            selected_columns = keep + [c for c in selected_columns if c not in keep]
            if not keep:
                good_cols = [selected_columns[0]]
                k = 1
            else:
                good_cols = keep
                k = len(keep)
            if len(selected_columns) > k:
                correlations = df_scaled[selected_columns].corr()
                for i, c in enumerate(selected_columns[k:], k):
                    # only take features that are somewhat uncorrelated with the rest
                    if np.max(np.abs(correlations[c].ravel()[:i])) < 0.9:
                        good_cols.append(c)
            if verbose > 0:
                print("[featsel] %i features after correlation filtering" % len(good_cols))

    # perform noise filtering on these features
    good_cols = _noise_filtering(df_scaled[good_cols].to_numpy(), target_scaled, good_cols, problem_type)
    if verbose > 0:
        print("[featsel] %i features after noise filtering" % len(good_cols))
        if not good_cols:
            print("[featsel] WARNING: Not a single good features was found...")

    # add keep columns back in
    good_cols = keep + [c for c in good_cols if c not in keep]
    if verbose > 0 and keep:
        print("[featsel] %i final features selected (including %i original keep features)." % (len(good_cols), len(keep)))
    return good_cols


class FeatureSelector(BaseEstimator):

    def __init__(
        self,
        problem_type="regression",
        featsel_runs=5,
        keep=None,
        n_jobs=1,
        verbose=0,
    ):
        """
        multi-step cross-validated feature selection

        Inputs:
            - problem_type: str, either "regression" or "classification" (default: "regression")
            - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
            - keep: list of features that should be kept no matter what
            - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
            - verbose: verbosity level (int; default: 0)

        Attributes:
            - good_cols_: list of good features (to select via pandas DataFrame columns)
            - original_columns_: original columns of X when calling fit
            - return_df_: whether fit was called with a dataframe in which case a df will be returned as well,
                          otherwise a numpy array
        """
        self.problem_type = problem_type
        self.featsel_runs = featsel_runs
        self.keep = keep
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """
        Selects features.

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        """
        self.return_df_ = isinstance(X, pd.DataFrame)
        # store column names as they'll be lost in the other check
        # first calling np.array assures that all the column names have the same dtype
        # as otherwise we get problems when calling np.random.permutation on the columns
        cols = list(np.array(list(X.columns))) if isinstance(X, pd.DataFrame) else []
        # check input variables
        X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression")
        if not cols:
            cols = ["x%i" % i for i in range(X.shape[1])]
        self.original_columns_ = cols
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # do the feature selection
        self.good_cols_ = select_features(df, target, self.featsel_runs, self.keep, self.problem_type, self.n_jobs, self.verbose)
        return self

    def transform(self, X):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        Returns:
            - new_X: new pandas dataframe or numpy array with only the good features
        """
        check_is_fitted(self, ["good_cols_"])
        if not self.good_cols_:
            if self.verbose > 0:
                print("[FeatureSelector] WARNING: No good features found; returning data unchanged.")
            return X
        # store column names as they'll be lost in the other check
        # first calling np.array assures that all the column names have the same dtype
        # as otherwise we get problems when calling np.random.permutation on the columns
        cols = list(np.array(list(X.columns))) if isinstance(X, pd.DataFrame) else []
        # check input variables
        X = check_array(X, force_all_finite="allow-nan")
        if not cols:
            cols = ["x%i" % i for i in range(X.shape[1])]
        if not cols == self.original_columns_:
            raise ValueError("[FeatureSelector] Not the same features as when calling fit.")
        # transform X into a dataframe (again) and select columns
        new_X = pd.DataFrame(X, columns=cols)[self.good_cols_]
        # possibly transform into a numpy array
        if not self.return_df_:
            new_X = new_X.to_numpy()
        return new_X

    def fit_transform(self, X, y):
        """
        Selects features and returns only those selected.

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        Returns:
            - new_X: new pandas dataframe or numpy array with only the good features
        """
        self.fit(X, y)
        return self.transform(X)
