from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range, object

import warnings
from collections import Counter
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn.linear_model as lm
from sympy.utilities.autowrap import ufuncify
import pint

from .feateng import generate_features, n_cols_generated
from .featsel import select_features


def _parse_units(units, ureg=None):
    """
    Convert a dict with string units to pint quantities.

    Inputs:
        - units: dict with {"variable_name": "unit"}
        - ureg: optional: a pint UnitRegistry

    Returns
        - parsed_units: dict with {"variable_name": pint Quantity}
    """
    if ureg is None:
        ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
    parsed_units = {}
    for c in units:
        try:
            parsed_units[c] = ureg.parse_expression(units[c])
        except pint.UndefinedUnitError as e:
            print("[AutoFeatRegression] WARNING: unit %r of column %r was not recognized and will be ignored!" % (units[c], c))
            parsed_units[c] = ureg.parse_expression("")
        parsed_units[c].__dict__["_magnitude"] = 1.
    return parsed_units


class AutoFeatRegression(object):

    def __init__(
        self,
        categorical_cols=[],
        feateng_cols=None,
        units={},
        feateng_steps=3,
        featsel_runs=5,
        max_gb=None,
        transformations=["exp", "log", "abs", "sqrt", "^2", "^3", "1/"],
        n_jobs=1,
    ):
        """
        multi-step feature engineering and cross-validated feature selection to generate promising additional
        features for your dataset and train a Lasso regression model with them.

        Inputs:
            - categorical_cols: list of column names of categorical features; these will be transformed into
                                0/1 encoding and not used in the feature engineering part (default: [])
            - feateng_cols: list of column names that should be used for the feature engineering part
                            (default None --> all except categorical_cols)
            - units: dictionary with {col_name: unit} where unit is a string that can be converted into a pint unit.
                     all columns without units are dimensionless and can be combined with any other column
                     (default: {} --> all columns are dimensionless).
            - feateng_steps: number of steps to perform in the feature engineering part (int; default: 3)
            - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
            - max_gb: if an int is given: maximum number of gigabytes to use in the process (i.e. mostly the
                      feature engineering part). this is no guarantee! it will lead to subsampling of the
                      data points if the new dataframe generated is n_rows * n_cols * 32bit > max_gb
            - transformations: list of transformations that should be applied; possible elements:
                               "exp", "log", "abs", "sqrt", "^2", "^3", "1/", "1+", "1-", "sin", "cos", "exp-", "2^"
                               (first 7, i.e., up to 1/, are applied by default)
            - n_jobs: how many jobs to run when selecting the features in parallel (default: 1)

        Note: when giving categorical_cols or feateng_cols, X later (i.e. when calling fit/fit_transform) has to be a DataFrame
        """
        self.categorical_cols = categorical_cols
        self.feateng_cols = feateng_cols
        self.units = units
        self.feateng_steps = feateng_steps
        self.max_gb = max_gb
        self.featsel_runs = featsel_runs
        self.transformations = transformations
        self.n_jobs = n_jobs
        # sympy formulas to generate new features
        self.feature_formulas = {}
        # compiled feature functions with columns
        self.feature_functions = {}
        # list of good new features that should be generated when calling transform()
        self.new_feat_cols = []
        # trained regression model
        self.regression_model = None

    def __getstate__(self):
        """
        get dict for pickling without feature_functions as they are not pickleable
        """
        return {k: self.__dict__[k] if k != "feature_functions" else {} for k in self.__dict__}

    def _transform_categorical_cols(self, df, cols={}):
        """
        Transform categorical features into 0/1 encoding.

        Inputs:
            - df: pandas dataframe with original features
            - cols: set of columns, from which the categorical_cols will be removed (optional)
        Returns:
            - df: dataframe with categorical features transformed into multiple 0/1 columns
            - if cols was given: updated cols set
        """
        if self.categorical_cols:
            e = OneHotEncoder(sparse=False, categories='auto')
            for c in self.categorical_cols:
                if cols:
                    cols.remove(c)
                ohe = e.fit_transform(df[c].values[:, None])
                df = df.join(pd.DataFrame(ohe, columns=["%s_%r" % (str(c), i) for i in e.categories_[0]], index=df.index))
            # remove the categorical column from our columns to consider
            df.drop(columns=self.categorical_cols, inplace=True)
        if cols:
            return df, cols
        return df

    def _apply_pi_theorem(self, df):
        if self.units:
            ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
            parsed_units = _parse_units(self.units, ureg)
            # use only original features
            parsed_units = {c: parsed_units[c] for c in self.feateng_cols if not parsed_units[c].dimensionless}
            print("[AutoFeatRegression] Applying the Pi Theorem")
            pi_theorem_results = ureg.pi_theorem(parsed_units)
            for i, r in enumerate(pi_theorem_results, 1):
                print("Pi Theorem %i: " % i, pint.formatter(r.items()))
                # compute the final result by multiplying and taking the power of
                cols = sorted(r)
                ptr = df[cols[0]].values**r[cols[0]]
                for c in cols[1:]:
                    ptr *= df[c].values**r[c]
                df["PT%i: %s" % (i, pint.formatter(r.items()).replace(" ", ""))] = ptr
        return df

    def _generate_features(self, df, new_feat_cols):
        """
        Generate additional features based on the feature formulas for all data points in the df.
        Only works after the model was fitted.

        Inputs:
            - df: pandas dataframe with original features
            - new_feat_cols: names of new features that should be generated (keys of self.feature_formulas)
        Returns:
            - df: dataframe with the additional feature columns added
        """
        assert new_feat_cols[0] in self.feature_formulas,\
            "[AutoFeatRegression] First call fit or fit_transform to generate the features!"
        print("[AutoFeatRegression] Computing %i new features." % len(new_feat_cols))
        # generate all good feature; unscaled this time
        feat_array = np.zeros((len(df), len(new_feat_cols)))
        for i, expr in enumerate(new_feat_cols):
            print("[AutoFeatRegression] %5i/%5i" % (i, len(new_feat_cols)), end="\r")
            if expr not in self.feature_functions:
                # generate a substitution expression based on all the original symbols of the original features
                # for the given generated feature in good cols
                # since sympy can handle only up to 32 original features in ufunctify, we need to check which features
                # to consider here, therefore perform some crude check to limit the number of features used
                cols = [c for c in self.feateng_cols if c in expr]
                try:
                    f = ufuncify((self.feature_formulas[c] for c in cols), self.feature_formulas[expr])
                except:
                    print("[AutoFeatRegression] Error while processing expression: %r" % expr)
                    raise
                self.feature_functions[expr] = (cols, f)
            else:
                cols, f = self.feature_functions[expr]
            try:
                feat_array[:, i] = f(*(df[c].values for c in cols))
            except RuntimeWarning as e:
                print("[AutoFeatRegression] Problem while evaluating expression: %r with columns %r - are maybe some values 0 that shouldn't be?" % (expr, cols))
                raise
        print("[AutoFeatRegression] %5i/%5i ...done." % (len(new_feat_cols), len(new_feat_cols)))
        df = df.join(pd.DataFrame(feat_array, columns=new_feat_cols, index=df.index))
        return df

    def fit_transform(self, X, y):
        """
        Fits the regression model and returns a new dataframe with the additional features.

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with one target variable for all n_datapoints
        Returns:
            - new_df: new pandas dataframe with all the original features (except categorical features transformed
                      into multiple 0/1 columns) and the most promising engineered features. This df can then be
                      used to train your final model.

        Please ensure that X only contains valid feature columns (including possible categorical variables).

        Note: we strongly encourage you to name your features X1 ...  Xn or something simple like this before passing
              a DataFrame to this model. This can help avoid potential problems with sympy later on.
        """
        # we need y as a numpy array later anyways
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.core.series.Series):
            target = y.values
        else:
            target = y
        if len(target.shape) > 1:
            target = target[:, 0]
        # possibly transform X to a dataframe or copy to mess around with it
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=["x%i" % i for i in range(10, 10 + X.shape[1])])
        else:
            df = X.copy()
            # in case X was created by filtering out some rows
            df.reset_index(drop=True, inplace=True)
        # possibly convert categorical columns
        cols = set(df.columns)
        df, cols = self._transform_categorical_cols(df, cols)
        # if we're not given specific feateng_cols, then just take all columns except categorical
        if not self.feateng_cols:
            self.feateng_cols = sorted(cols)
        # convert units to proper pint units
        if self.units:
            # need units for only and all feateng columns
            self.units = {c: self.units[c] if c in self.units else "" for c in self.feateng_cols}
            # apply pi-theorem -- additional columns are not used for regular feature engineering (for now)!
            df = self._apply_pi_theorem(df)
        # subsample data points and targets in case we'll generate too many features
        # (n_rows * n_cols * 32/8)/1000000000 <= max_gb
        n_cols = n_cols_generated(len(self.feateng_cols), self.feateng_steps, len(self.transformations))
        n_gb = (len(df) * n_cols) / 250000000
        print("[AutoFeatRegression] The %i step feature engineering process could generate up to %i features." % (self.feateng_steps, n_cols))
        print("[AutoFeatRegression] With %i data points this new feature matrix would use about %.2f gb of space." % (len(df), n_gb))
        if self.max_gb and n_gb > self.max_gb:
            n_rows = int(self.max_gb * 250000000 / n_cols)
            print("[AutoFeatRegression] As you specified a limit of %.1d gb, the number of data points is subsampled to %i" % (self.max_gb, n_rows))
            subsample_idx = np.random.permutation(list(df.index))[:n_rows]
            df_subs = df.iloc[subsample_idx]
            df_subs.reset_index(drop=True, inplace=True)
            target_sub = target[subsample_idx]
        else:
            df_subs = df
            target_sub = target
        # generate features and scale to have 0 mean and unit std
        df_scaled, self.feature_formulas = generate_features(df_subs, self.feateng_cols, _parse_units(self.units),
                                                             self.feateng_steps, self.transformations)
        s = StandardScaler()
        df_scaled = pd.DataFrame(s.fit_transform(df_scaled), columns=df_scaled.columns, dtype=np.float32)
        # do sort of a cross-validation (i.e., randomly subsample data points)
        # to select features with high relevance
        selected_columns = []
        idx = list(df_scaled.index)
        print("[AutoFeatRegression] Selecting good features in %i runs" % self.featsel_runs)

        # select good features in 5 runs in parallel
        def run_select_features(i):
            np.random.seed(i)
            train_idx = np.random.permutation(idx)[:int(0.8 * len(idx))]
            return select_features(df_scaled.iloc[train_idx], target_sub[train_idx], True, eps=1e-8)
        if self.n_jobs == 1:
            # only use parallelization code if you actually parallelize
            selected_columns = []
            for i in range(self.featsel_runs):
                selected_columns.extend(run_select_features(i))
        else:
            flatten_lists = lambda l: [item for sublist in l for item in sublist]
            selected_columns = flatten_lists(Parallel(n_jobs=self.n_jobs, verbose=100)(delayed(run_select_features)(i) for i in range(self.featsel_runs)))
        # check in how many runs each feature was selected and only takes those that were selected in more than one run
        selected_columns = Counter(selected_columns)
        original_features = list(df.columns)
        good_cols = [c for c in selected_columns if selected_columns[c] > 1 and c not in original_features]
        print("[AutoFeatRegression] %i features occurred in more than one featsel run." % len(good_cols))
        # train another regression model on these features
        df_scaled = df_scaled[original_features + good_cols]
        X = df_scaled.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = lm.LassoLarsCV(eps=1e-8)
            reg.fit(X, target_sub)
        weights = dict(zip(list(df_scaled.columns), reg.coef_))
        good_cols = [c for c in weights if abs(weights[c]) >= 1e-5 and c not in original_features]
        print("[AutoFeatRegression] %i new features selected." % len(good_cols))
        # re-generate all good feature again; unscaled this time
        df = self._generate_features(df, good_cols)
        print("[AutoFeatRegression] Training final regression model.")
        # train final regression model all selected features (twice)
        for i in range(2):
            X = df.values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg = lm.LassoLarsCV(eps=1e-16)
                reg.fit(X, target)
            weights = dict(zip(list(df.columns), reg.coef_))
            if not i:
                self.new_feat_cols = [c for c in weights if abs(weights[c] * df[c].std()) >= 1e-6 and c not in original_features]
                df = df[original_features + self.new_feat_cols]
        # filter out unnecessary junk from self.feature_formulas
        self.feature_formulas = {f: self.feature_formulas[f] for f in self.new_feat_cols + self.feateng_cols}
        self.feature_functions = {f: self.feature_functions[f] for f in self.new_feat_cols}
        print("[AutoFeatRegression] Trained model coefficients:")
        print(reg.intercept_)
        for c in sorted(weights, key=lambda x: abs(weights[x]), reverse=True):
            if abs(weights[c]) < 1e-5:
                break
            print("%.6f * %s" % (weights[c], c))
        print("[AutoFeatRegression] Final R^2: %.4f" % reg.score(X, target))
        self.regression_model = reg
        # final dataframe contains original columns, good additional columns, and target column
        print("[AutoFeatRegression] Final dataframe with %i feature columns (%i new)." % (len(df.columns), len(df.columns) - len(original_features)))
        return df

    def fit(self, X, y):
        print("[AutoFeatRegression] Warning: This just calls fit_transform() but does not return the transformed dataframe.")
        print("[AutoFeatRegression] It is much more efficient to call fit_transform() instead of fit() and transform()!")
        _ = self.fit_transform(X, y)

    def transform(self, X):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        Returns:
            - new_df: new pandas dataframe with all the original features (except categorical features transformed
                      into multiple 0/1 columns) and the most promising engineered features. This df can then be
                      used to train your final model.
        """
        assert self.regression_model is not None, "[AutoFeatRegression] First call fit or fit_transform to train the model!"
        # possibly transform X to a dataframe or copy to mess around with it
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=["x%i" % i for i in range(10, 10 + X.shape[1])])
        else:
            df = X.copy()
        # possibly convert categorical columns
        df = self._transform_categorical_cols(df)
        # possibly apply pi-theorem
        df = self._apply_pi_theorem(df)
        # generate engineered features
        df = self._generate_features(df, self.new_feat_cols)
        return df

    def predict(self, X):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
        Returns:
            - y_pred: vector of predicted targets return by regression_model.predict()
        """
        assert self.regression_model is not None, "[AutoFeatRegression] First call fit or fit_transform to train the model!"
        # possibly transform X to a dataframe or copy to mess around with it
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=["x%i" % i for i in range(10, 10 + X.shape[1])])
        else:
            df = X
        # check if the dataframe was already transformed
        if not self.new_feat_cols[0] in df:
            df = self.transform(df)
        return self.regression_model.predict(df.values)

    def score(self, X, y):
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with one target variable for all n_datapoints
        Returns:
            - R^2 value returned by regression_model.score()
        """
        assert self.regression_model is not None, "[AutoFeatRegression] First call fit or fit_transform to train the model!"
        # possibly transform X to a dataframe or copy to mess around with it
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=["x%i" % i for i in range(10, 10 + X.shape[1])])
        else:
            df = X
        # check if the dataframe was already transformed
        if not self.new_feat_cols[0] in df:
            df = self.transform(df)
        # get the target as an array
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.core.series.Series):
            y = y.values
        return self.regression_model.score(df.values, y)
