# -*- coding: utf-8 -*-
# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import zip
import warnings
from collections import Counter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm


def select_features_1run(df, target, max_it=100, eps=1e-16, verbose=0):
    """
    Inputs:
        - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
              to the n training data points. The variables have to be scaled to have 0 mean and unit variance.
        - target: n dimensional array with targets corresponding to the data points in df
        - max_it: how many iterations will be performed at most (int; default: 100)
        - eps: eps parameter for LassoLarsCV regression model (float; default: 1e-16;
               might need to increase that to ~1e-8 or 1e-5 if you get a warning)
        - verbose: verbosity level (int; default: 0)
    Returns:
        - good_cols: list of column names for df with which a regression model can be trained
    """
    # split in training and test parts
    df_train = df[:max(3, int(0.8 * len(df)))]
    df_test = df[int(0.8 * len(df)):]
    target_train = target[:max(3, int(0.8 * len(df)))]
    target_test = target[int(0.8 * len(df)):]
    if not (len(df_train) == len(target_train) and len(df_test) == len(target_test)):
        raise ValueError("[featsel] df and target dimension mismatch.")

    scaler = StandardScaler()
    # good cols contains the currently considered good features (=columns)
    good_cols = []
    best_cols = []
    # we want to select up to thr features (how much a regression model is comfortable with)
    thr = int(0.5 * df_train.shape[0])
    # our first target is the original target variable; later we operate on (target - predicted_target)
    new_target = target_train
    residual = np.mean(np.abs(target_test))
    last_residuals = np.zeros(max_it)
    smallest_residual = 10. * residual
    it = 0
    # we try optimizing features until we have converged or run over max_it
    while (it < max_it) and (not np.sum(np.isclose(residual, last_residuals)) >= 2):
        if verbose and not it % 10:
            print("[featsel] Iteration %3i; %3i selected features with residual: %.6f" % (it, len(good_cols), residual))
        last_residuals[it] = residual
        it += 1
        # select new possibly good columns from all but the currently considered good columns
        cols = set(df_train.columns)
        cols.difference_update(good_cols)
        cols_list = list(cols)
        # compute the absolute correlation of the (scaled) features with the (scaled) target variable
        w = np.abs(np.dot(scaler.fit_transform(new_target[:, None])[:, 0], df_train[cols_list].to_numpy()))
        # add promising features such that len(previous good cols + new cols) = thr
        good_cols.extend([cols_list[c] for c in np.argsort(w)[-(thr - len(good_cols)):]])
        # compute the regression residual based on the best features so far
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = lm.LassoLarsCV(eps=eps)
            X = df_train[good_cols].to_numpy()
            reg.fit(X, target_train)
        new_target = target_train - reg.predict(X)
        residual = np.mean(np.abs(target_test - reg.predict(df_test[good_cols].to_numpy())))
        # update the good columns based on the regression coefficients
        weights = dict(zip(good_cols, reg.coef_))
        good_cols = [c for c in weights if abs(weights[c]) > 1e-6]
        if residual < smallest_residual:
            smallest_residual = residual
            best_cols = [c for c in good_cols]
    if verbose:
        print("[featsel] Iteration %3i; %3i selected features with residual: %.6f  --> done." % (it, len(best_cols), smallest_residual))
    return best_cols


def select_features(df, target, featsel_runs=5, max_it=100, n_jobs=1, verbose=0):
    """
    Inputs:
        - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
              to the n training data points.
        - target: n dimensional array with targets corresponding to the data points in df
        - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
        - max_it: how many iterations will be performed at most (int; default: 100)
        - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
        - verbose: verbosity level (int; default: 0)
    Returns:
        - good_cols: list of column names for df with which a regression model can be trained
    """
    # scale features to have 0 mean and unit std
    if verbose:
        if featsel_runs > df.shape[0]:
            print("[featsel] WARNING: Less data points than featsel runs!!")
        print("[featsel] Scaling data...", end="")
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, dtype=np.float32)
    if verbose:
        print("done.")

    # select good features in 5 runs in parallel
    # by doing sort of a cross-validation (i.e., randomly subsample data points)
    def run_select_features(i):
        np.random.seed(i)
        rand_idx = np.random.permutation(df.index)
        return select_features(df.iloc[rand_idx], target[rand_idx], max_it=max_it, eps=1e-8, verbose=verbose)
    if n_jobs == 1:
        # only use parallelization code if you actually parallelize
        selected_columns = []
        for i in range(featsel_runs):
            selected_columns.extend(run_select_features(i))
    else:
        def flatten_lists(l):
            return [item for sublist in l for item in sublist]

        selected_columns = flatten_lists(Parallel(n_jobs=n_jobs, verbose=100)(delayed(run_select_features)(i) for i in range(featsel_runs)))

    # check in how many runs each feature was selected and only takes those that were selected in more than one run
    selected_columns = Counter(selected_columns)
    good_cols = [c for c in selected_columns if selected_columns[c] > 1]
    if verbose:
        print("[featsel] %i features occurred in more than one featsel run." % len(good_cols))
    # train another regression model on these features
    df = df[good_cols]
    X = df.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = lm.LassoLarsCV(eps=1e-8)
        reg.fit(X, target)
    weights = dict(zip(list(df.columns), reg.coef_))
    good_cols = [c for c in weights if abs(weights[c]) >= 1e-6]
    if verbose:
        print("[featsel] %i new features selected." % len(good_cols))
    return good_cols
