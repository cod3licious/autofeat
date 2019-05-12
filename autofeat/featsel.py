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
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.feature_selection import SelectFdr, f_regression
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm


def _select_features_1run(df, target, max_it=150, eps=1e-16, verbose=0):
    """
    Inputs:
        - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
              to the n training data points. The variables have to be scaled to have 0 mean and unit variance.
        - target: n dimensional array with targets corresponding to the data points in df
        - max_it: how many iterations will be performed at most (int; default: 150)
        - eps: eps parameter for LassoLarsCV regression model (float; default: 1e-16;
               might need to increase that to ~1e-8 or 1e-5 if you get a warning)
        - verbose: verbosity level (int; default: 0)
    Returns:
        - good_cols: list of column names for df with which a regression model can be trained
    """
    scaler = StandardScaler()
    # good cols contains the currently considered good features (=columns)
    good_cols = []
    best_cols = []
    # shuffle and split data in training and test parts
    df_train, df_test, target_train, target_test = train_test_split(df, target, test_size=0.3, random_state=max_it)
    if df_train.shape[0] < 10:
        df_train, df_test, target_train, target_test = df, df, target, target
    # we want to select up to thr features (how much a regression model is comfortable with)
    thr = max(1, int(0.5 * df_train.shape[0]))
    # our first target is the original target variable; later we operate on (target - predicted_target)
    new_target = target_train
    r2 = -1e15
    last_r2s = -1e10*np.ones(max_it)
    best_r2 = 10. * r2
    it = 0
    # we try optimizing features until we have converged or run over max_it
    while (it < max_it) and (not (np.sum(np.isclose(r2, last_r2s)) >= 2) or r2 >= 0.99999):
        if verbose > 0 and not it % 10:
            print("[featsel] Iteration %3i; %3i selected features with R^2: %.6f" % (it, len(good_cols), r2))
        last_r2s[it] = r2
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
        # compute the r2 score of the model on the test data
        r2 = reg.score(df_test[good_cols].to_numpy(), target_test)
        # shuffle the data again for the next iteration
        if df.shape[0] > 14:
            df_train, df_test, target_train, target_test = train_test_split(df, target, test_size=0.3, random_state=it)
        # and compute a new target
        new_target = target_train - reg.predict(df_train[good_cols].to_numpy())
        # update the good columns based on the regression coefficients
        weights = dict(zip(good_cols, reg.coef_))
        good_cols = [c for c in weights if abs(weights[c]) > 5e-3]
        if r2 > best_r2:
            best_r2 = r2
            best_cols = [c for c in good_cols]
    # last model on the whole df to filter the best columns
    X = df[best_cols].to_numpy()
    if df.shape[0] > 50:
        rand_noise = np.random.permutation(X.flatten()).reshape(X.shape)
        X = np.hstack([X, rand_noise])
    rand_noise = np.random.randn(df.shape[0], max(3, int(0.5*len(best_cols))))
    X = np.hstack([X, rand_noise])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = lm.LassoLarsCV(eps=eps)
        reg.fit(X, target)
    weights = dict(zip(best_cols, reg.coef_[:len(best_cols)]))
    # only include features that are more important than our known noise features
    w_thr = np.max(np.abs(reg.coef_[len(best_cols):]))
    best_cols = [c for c in weights if abs(weights[c]) > w_thr]
    if verbose > 0:
        print("[featsel] Iteration %3i; %3i selected features with R^2: %.6f  --> done." % (it, len(best_cols), best_r2))
    return best_cols


def select_features(df, target, featsel_runs=3, max_it=150, w_thr=1e-4, keep=None, n_jobs=1, verbose=0):
    """
    Inputs:
        - df: nxp pandas DataFrame with n data points and p features; to avoid overfitting, only provide data belonging
              to the n training data points.
        - target: n dimensional array with targets corresponding to the data points in df
        - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 3)
        - max_it: how many iterations will be performed at most (int; default: 150)
        - w_thr: threshold on the final Lasso model weights to filter the features (float; default: 1e-4)
        - keep: list of features that should be kept no matter what
        - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
        - verbose: verbosity level (int; default: 0)
    Returns:
        - good_cols: list of column names for df with which a regression model can be trained
    """
    if not (len(df) == len(target)):
        raise ValueError("[featsel] df and target dimension mismatch.")
    if keep is None:
        keep = []
    # scale features to have 0 mean and unit std
    if verbose > 0:
        if featsel_runs > df.shape[0]:
            print("[featsel] WARNING: Less data points than featsel runs!!")
        print("[featsel] Scaling data...", end="")
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, dtype=np.float32)
        target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()
    if verbose > 0:
        print("done.")

    # quick and dirty univariate filtering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fsel = SelectFdr(f_regression, alpha=0.1).fit(df_scaled, target_scaled)
        cols = keep + [df_scaled.columns[i] for i in fsel.get_support(True) if df_scaled.columns[i] not in keep]
        if cols:
            df_scaled = df_scaled[cols]
            if verbose > 0:
                print("[featsel] %i/%i features after univariate filtering" % (len(df_scaled.columns), len(fsel.get_support())))

    # select good features in k runs in parallel
    # by doing sort of a cross-validation (i.e., randomly subsample data points)
    def run_select_features(i):
        if verbose > 0:
            print("[featsel] Feature selection run %i/%i" % (i+1, featsel_runs))
        np.random.seed(i)
        rand_idx = np.random.permutation(df_scaled.index)[:max(10, int(0.8 * len(df_scaled)))]
        return _select_features_1run(df_scaled.iloc[rand_idx], target_scaled[rand_idx], max_it=max_it, eps=1e-8, verbose=verbose-1)
    good_cols = [c for c in keep]
    if featsel_runs >= 1:
        if n_jobs == 1:
            # only use parallelization code if you actually parallelize
            selected_columns = []
            for i in range(featsel_runs):
                selected_columns.extend(run_select_features(i))
        else:
            def flatten_lists(l):
                return [item for sublist in l for item in sublist]

            selected_columns = flatten_lists(Parallel(n_jobs=n_jobs, verbose=100*verbose)(delayed(run_select_features)(i) for i in range(featsel_runs)))

        if len(selected_columns) > 1:
            selected_columns = Counter(selected_columns)
            selected_columns = sorted(selected_columns, key=selected_columns.get, reverse=True)
            selected_columns = keep + [c for c in selected_columns if c not in keep]
            if verbose > 0:
                print("[featsel] %i features after %i feature selection runs" % (len(selected_columns), featsel_runs))
            correlations = df_scaled[selected_columns].corr()
            if not keep:
                good_cols.append(selected_columns[0])
                k = 1
            else:
                k = len(keep)
            for i, c in enumerate(selected_columns[k:], k):
                # only take features that are somewhat uncorrelated with the rest
                if np.max(np.abs(correlations[c].ravel()[:i])) < 0.9:
                    good_cols.append(c)
            if verbose > 0:
                print("[featsel] %i features after correlation filtering" % len(good_cols))
        else:
            good_cols += selected_columns
    if not good_cols:
        good_cols = list(df.columns)
    # perform recursive feature elimination on these features
    df_scaled = df_scaled[good_cols]
    X = df_scaled.to_numpy()
    if df_scaled.shape[0] > 50:
        rand_noise = np.random.permutation(X.flatten()).reshape(X.shape)
        X = np.hstack([X, rand_noise])
    rand_noise = np.random.randn(df.shape[0], max(3, int(0.5*len(good_cols))))
    X = np.hstack([X, rand_noise])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = lm.LassoLarsCV(eps=1e-16)
        reg.fit(X, target)
    weights = dict(zip(good_cols, reg.coef_[:len(good_cols)]))
    # only include features that are more important than our known noise features
    noise_w_thr = np.max(np.abs(reg.coef_[len(good_cols):]))
    good_cols = [c for c in weights if abs(weights[c]) > noise_w_thr]
    if verbose > 0:
        print("[featsel] %i features after noise filtering" % len(good_cols))
    if not good_cols:
        if verbose > 0:
            print("[featsel] WARNING: Not a single good features was found...")
        return keep
    # train again a regression model, but this time on the original (unscaled) data
    df = df[good_cols]
    X = df.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = lm.LassoLarsCV(eps=1e-16)
        reg.fit(X, target)
        # alphas in CV are generally chosen a bit too small
        reg = lm.LassoLars(alpha=1.5*reg.alpha_, eps=1e-16)
        reg.fit(X, target)
    weights = dict(zip(list(df.columns), reg.coef_))
    good_cols = [c for c in sorted(weights, key=lambda x: abs(weights[x]), reverse=True) if abs(weights[c] * df[c].std()) >= w_thr]
    # add keep columns back in
    good_cols = keep + [c for c in good_cols if c not in keep]
    if verbose > 0:
        if not good_cols:
            print("[featsel] WARNING: Not a single good features was found...")
        print("[featsel] %i final features selected (including %i original keep features)." % (len(good_cols), len(keep)))
    return good_cols


class FeatureSelector(BaseEstimator):

    def __init__(
        self,
        featsel_runs=3,
        max_it=150,
        w_thr=1e-6,
        keep=None,
        n_jobs=1,
        verbose=0,
    ):
        """
        multi-step cross-validated feature selection

        Inputs:
            - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 3)
            - max_it: maximum number of iterations for the feature selection (int; default 150)
            - w_thr: threshold on the final Lasso model weights to filter the features (float; default: 1e-6)
            - keep: list of features that should be kept no matter what
            - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
            - verbose: verbosity level (int; default: 0)

        Attributes:
            - good_cols_: list of good features (to select via pandas DataFrame columns)
            - original_columns_: original columns of X when calling fit
            - return_df_: whether fit was called with a dataframe in which case a df will be returned as well,
                          otherwise a numpy array
        """
        self.featsel_runs = featsel_runs
        self.max_it = max_it
        self.w_thr = w_thr
        self.keep = keep
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """
        Selects features.

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with one target variable for all n_datapoints
        """
        self.return_df_ = isinstance(X, pd.DataFrame)
        # store column names as they'll be lost in the other check
        cols = list(X.columns) if isinstance(X, pd.DataFrame) else []
        # check input variables
        X, target = check_X_y(X, y, y_numeric=True)
        if not cols:
            cols = ["x%i" % i for i in range(X.shape[1])]
        self.original_columns_ = cols
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # do the feature selection
        self.good_cols_ = select_features(df, target, self.featsel_runs, self.max_it, self.w_thr, self.keep, self.n_jobs, self.verbose)
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
        cols = list(X.columns) if isinstance(X, pd.DataFrame) else []
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
            - y: pandas dataframe or numpy array with one target variable for all n_datapoints
        Returns:
            - new_X: new pandas dataframe or numpy array with only the good features
        """
        self.fit(X, y)
        return self.transform(X)
