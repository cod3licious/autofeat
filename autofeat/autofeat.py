# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

from __future__ import annotations

import logging
import warnings

import numba as nb
import numpy as np
import pandas as pd
import pint
import sklearn.linear_model as lm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sympy.utilities.lambdify import lambdify

from autofeat.feateng import colnames2symbols, engineer_features, n_cols_generated
from autofeat.featsel import select_features

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def _parse_units(units: dict, ureg: pint.UnitRegistry | None = None, verbose: int = 0):
    """
    Convert a dict with string units to pint quantities.

    Inputs:
        - units: dict with {"variable_name": "unit"}
        - ureg: optional: a pint UnitRegistry
        - verbose: verbosity level (int; default: 0)

    Returns:
        - parsed_units: dict with {"variable_name": pint Quantity}
    """
    parsed_units = {}
    if units:
        if ureg is None:
            ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
        for c in units:
            try:
                parsed_units[c] = ureg.parse_expression(units[c])
            except pint.UndefinedUnitError:
                if verbose > 0:
                    logging.warning(f"[AutoFeat] unit {units[c]} of column {c} was not recognized and will be ignored!")
                parsed_units[c] = ureg.parse_expression("")
            parsed_units[c].__dict__["_magnitude"] = 1.0
    return parsed_units


class AutoFeatModel(BaseEstimator):
    def __init__(
        self,
        problem_type: str = "regression",
        categorical_cols: list | None = None,
        feateng_cols: list | None = None,
        units: dict | None = None,
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        max_gb: int | None = None,
        transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
        apply_pi_theorem: bool = True,
        always_return_numpy: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        """
        multi-step feature engineering and cross-validated feature selection to generate promising additional
        features for your dataset and train a linear prediction model with them.

        Inputs:
            - problem_type: str, either "regression" or "classification" (default: "regression")
            - categorical_cols: list of column names of categorical features; these will be transformed into
                                0/1 encoding (default: None)
            - feateng_cols: list of column names that should be used for the feature engineering part
                            (default None --> all, with categorical_cols in 0/1 encoding)
            - units: dictionary with {col_name: unit} where unit is a string that can be converted into a pint unit.
                     all columns without units are dimensionless and can be combined with any other column.
                     Note: it is assumed that all features are of comparable magnitude, i.e., not one variable is in
                           m and another in mm. If this needs to be accounted for, please scale your variables before
                           passing them to autofeat!
                     (default: None --> all columns are dimensionless).
            - feateng_steps: number of steps to perform in the feature engineering part (int; default: 2)
            - featsel_runs: number of times to perform in the feature selection part with a random fraction of data points (int; default: 5)
            - max_gb: if an int is given: maximum number of gigabytes to use in the process (i.e. mostly the
                      feature engineering part). this is no guarantee! it will lead to subsampling of the
                      data points if the new dataframe generated is n_rows * n_cols * 32bit > max_gb
                      Note: this is only an approximate estimate of the final matrix; intermediate representations could easily
                            take up at least 2 or 3 times that much space...If you can, subsample before, you know your data best.
            - transformations: list of transformations that should be applied; possible elements:
                               "1/", "exp", "log", "abs", "sqrt", "^2", "^3", "1+", "1-", "sin", "cos", "exp-", "2^"
                               (first 7, i.e., up to ^3, are applied by default)
            - apply_pi_theorem: whether or not to apply the pi theorem (if units are given; bool; default: True)
            - always_return_numpy: whether to always return a numpy array instead of a pd dataframe when calling (fit_)transform
                                   (default: False; mainly used for sklearn estimator checks)
            - n_jobs: how many jobs to run when selecting the features in parallel (int; default: 1)
            - verbose: verbosity level (int; default: 0)

        Attributes:
            - original_columns_: original columns of X when calling fit
            - all_columns_: columns of X after calling fit
            - categorical_cols_map_: dict mapping from the original categorical columns to a list with new column names
            - feateng_cols_: actual columns used for the feature engineering
            - feature_formulas_: sympy formulas to generate new features
            - feature_functions_: compiled feature functions with columns
            - new_feat_cols_: list of good new features that should be generated when calling transform()
            - good_cols_: columns selected in the feature selection process, used with the final prediction model
            - prediction_model_: sklearn model instance used for the predictions

        Note: when giving categorical_cols or feateng_cols, X later (i.e. when calling fit/fit_transform) has to be a DataFrame
        """
        self.problem_type = problem_type
        self.categorical_cols = categorical_cols
        self.feateng_cols = feateng_cols
        self.units = units
        self.feateng_steps = feateng_steps
        self.max_gb = max_gb
        self.featsel_runs = featsel_runs
        self.transformations = transformations
        self.apply_pi_theorem = apply_pi_theorem
        self.always_return_numpy = always_return_numpy
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __getstate__(self):
        """Get dict for pickling without feature_functions as they are not pickleable"""
        return {k: self.__dict__[k] if k != "feature_functions_" else {} for k in self.__dict__}

    def _transform_categorical_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features into 0/1 encoding.

        Inputs:
            - df: pandas dataframe with original features
        Returns:
            - df: dataframe with categorical features transformed into multiple 0/1 columns
        """
        self.categorical_cols_map_ = {}
        if self.categorical_cols:
            e = OneHotEncoder(sparse_output=False, categories="auto")
            for c in self.categorical_cols:
                if c not in df.columns:
                    raise ValueError(f"[AutoFeat] categorical_col {c} not in df.columns")
                ohe = e.fit_transform(df[c].to_numpy()[:, None])
                new_cat_cols = [f"cat_{c}_{i}" for i in e.categories_[0]]
                self.categorical_cols_map_[c] = new_cat_cols
                df = df.join(pd.DataFrame(ohe, columns=new_cat_cols, index=df.index))
            # remove the categorical column from our columns to consider
            df = df.drop(columns=self.categorical_cols)
        return df

    def _apply_pi_theorem(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.apply_pi_theorem and self.units:
            ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
            parsed_units = _parse_units(self.units, ureg, self.verbose)
            # use only original features
            parsed_units = {c: parsed_units[c] for c in self.feateng_cols_ if not parsed_units[c].dimensionless}
            if self.verbose:
                logging.info("[AutoFeat] Applying the Pi Theorem")
            pi_theorem_results = ureg.pi_theorem(parsed_units)
            for i, r in enumerate(pi_theorem_results, 1):
                if self.verbose:
                    logging.info(f"[AutoFeat] Pi Theorem {i}: ")
                    logging.info(pint.formatter(r.items()))
                # compute the final result by multiplying and taking the power of
                cols = sorted(r)
                # only use data points where non of the affected columns are NaNs
                not_na_idx = df[cols].notna().all(axis=1)
                ptr = df[cols[0]].to_numpy()[not_na_idx] ** r[cols[0]]
                for c in cols[1:]:
                    ptr *= df[c].to_numpy()[not_na_idx] ** r[c]
                df.loc[not_na_idx, f"PT{i}_{pint.formatter(r.items()).replace(' ', '')}"] = ptr
        return df

    def _generate_features(self, df: pd.DataFrame, new_feat_cols: list) -> pd.DataFrame:
        """
        Generate additional features based on the feature formulas for all data points in the df.
        Only works after the model was fitted.

        Inputs:
            - df: pandas dataframe with original features
            - new_feat_cols: names of new features that should be generated (keys of self.feature_formulas_)

        Returns:
            - df: dataframe with the additional feature columns added
        """
        check_is_fitted(self, ["feature_formulas_"])
        if not new_feat_cols:
            return df
        if new_feat_cols[0] not in self.feature_formulas_:
            raise RuntimeError("[AutoFeat] First call fit or fit_transform to generate the features!")
        if self.verbose:
            logging.info(f"[AutoFeat] Computing {len(new_feat_cols)} new features.")
        # generate all good feature; unscaled this time
        feat_array = np.zeros((len(df), len(new_feat_cols)))
        for i, expr in enumerate(new_feat_cols):
            if self.verbose:
                print(f"[AutoFeat] {i:5}/{len(new_feat_cols):5} new features", end="\r")
            if expr not in self.feature_functions_:
                # generate a substitution expression based on all the original symbols of the original features
                # for the given generated feature in good cols
                # since sympy can handle only up to 32 original features in ufunctify, we need to check which features
                # to consider here, therefore perform some crude check to limit the number of features used
                cols = [c for i, c in enumerate(self.feateng_cols_) if colnames2symbols(c, i) in expr]
                if not cols:
                    # this can happen if no features were selected and the expr is "E" (i.e. the constant e)
                    f = None
                    f_jit = None
                else:
                    try:
                        f = lambdify([self.feature_formulas_[c] for c in cols], self.feature_formulas_[expr])
                        f_jit = nb.njit(f)
                    except Exception:
                        logging.exception(f"[AutoFeat] Error while processing expression: {expr}")
                        raise
                self.feature_functions_[expr] = (cols, f, f_jit)
            else:
                cols, f, f_jit = self.feature_functions_[expr]
            if f is not None and f_jit is not None:
                # only generate features for completely not-nan rows
                not_na_idx = df[cols].notna().all(axis=1)
                try:
                    try:
                        feat = f_jit(*(df[c].to_numpy(dtype=float)[not_na_idx] for c in cols))
                    except nb.TypingError:
                        # lambified abs fn with non trivial inputs doesn't jit compile with numba, yet
                        # fallback on the non jitted version of the function
                        feat = f(*(df[c].to_numpy(dtype=float)[not_na_idx] for c in cols))
                        # henceforth, always use the non jitted version of the function
                        self.feature_functions_[expr] = (cols, f, f)
                    feat_array[not_na_idx, i] = feat
                    feat_array[~not_na_idx, i] = np.nan
                except RuntimeWarning:
                    logging.warning(
                        f"[AutoFeat] Problem while evaluating expression: {expr} with columns {cols}",
                        " - is the data in a different range then when calling .fit()? Are maybe some values 0 that shouldn't be?",
                    )
                    raise
        if self.verbose:
            logging.info(f"[AutoFeat] {len(new_feat_cols):5}/{len(new_feat_cols):5} new features ...done.")
        return df.join(pd.DataFrame(feat_array, columns=new_feat_cols, index=df.index))

    def _X2df(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Helper function that ensures correctness of the input data for classification tasks.
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)

        Returns:
            - df: transformed dataframe
        """
        # store column names as they'll be lost in the other check
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
        # check input variables
        X = check_array(X, dtype=None)
        if not cols:
            cols = [f"x{i:03}" for i in range(X.shape[1])]
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # do we need to call transform?
        if list(df.columns) != self.all_columns_:
            temp = self.always_return_numpy
            self.always_return_numpy = False
            df = self.transform(df)
            self.always_return_numpy = temp
        return df

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Fits the regression model and returns a new dataframe with the additional features.

        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        Returns:
            - new_df: new pandas dataframe with all the original features (except categorical features transformed
                      into multiple 0/1 columns) and the most promising engineered features. This df can then be
                      used to train your final model.

        Please ensure that X only contains valid feature columns (including possible categorical variables).

        Note: we strongly encourage you to name your features X1 ...  Xn or something simple like this before passing
              a DataFrame to this model. This can help avoid potential problems with sympy later on.
              The data should only contain finite values (no NaNs etc.)
        """
        # store column names as they'll be lost in the other check
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
        # check input variables
        X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression", dtype=None)
        if not cols:
            # the additional zeros in the name are because of the variable check in _generate_features,
            # where we check if the column name occurs in the the expression. this would lead to many
            # false positives if we have features x1 and x10...x19 instead of x001...x019.
            cols = [f"x{i:03}" for i in range(X.shape[1])]
        self.original_columns_ = cols
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # possibly convert categorical columns
        df = self._transform_categorical_cols(df)
        # if we're not given specific feateng_cols, then just take all columns except categorical
        if self.feateng_cols:
            fcols = []
            for c in self.feateng_cols:
                if c not in self.original_columns_:
                    raise ValueError(f"[AutoFeat] feateng_col {c} not in df.columns")
                if c in self.categorical_cols_map_:
                    fcols.extend(self.categorical_cols_map_[c])
                else:
                    fcols.append(c)
            self.feateng_cols_ = fcols
        else:
            self.feateng_cols_ = list(df.columns)
        # convert units to proper pint units
        if self.units:
            # need units for only and all feateng columns
            self.units = {c: self.units[c] if c in self.units else "" for c in self.feateng_cols_}
            # apply pi-theorem -- additional columns are not used for regular feature engineering (for now)!
            df = self._apply_pi_theorem(df)
        # subsample data points and targets in case we'll generate too many features
        # (n_rows * n_cols * 32/8)/1000000000 <= max_gb
        n_cols = n_cols_generated(len(self.feateng_cols_), self.feateng_steps, len(self.transformations))
        n_gb = (len(df) * n_cols) / 250000000
        if self.verbose:
            logging.info(
                f"[AutoFeat] The {self.feateng_steps} step feature engineering process could generate up to {n_cols} features.",
            )
            logging.info(
                f"[AutoFeat] With {len(df)} data points this new feature matrix would use about {n_gb:.2f} gb of space.",
            )
        if self.max_gb and n_gb > self.max_gb:
            n_rows = int(self.max_gb * 250000000 / n_cols)
            if self.verbose:
                logging.info(
                    f"[AutoFeat] As you specified a limit of {self.max_gb:.1f} gb, the number of data points is subsampled to {n_rows}",
                )
            subsample_idx = np.random.permutation(list(df.index))[:n_rows]
            df_subs = df.iloc[subsample_idx]
            df_subs = df_subs.reset_index(drop=True)
            target_sub = target[subsample_idx]
        else:
            df_subs = df.copy()
            target_sub = target.copy()
        # generate features
        df_subs, self.feature_formulas_ = engineer_features(
            df_subs,
            self.feateng_cols_,
            _parse_units(self.units, verbose=self.verbose),  # type: ignore
            self.feateng_steps,
            self.transformations,
            self.verbose,
        )
        # select predictive features
        if self.featsel_runs <= 0:
            if self.verbose:
                logging.warning("[AutoFeat] Not performing feature selection.")
            good_cols = df_subs.columns
        elif self.problem_type in ("regression", "classification"):
            good_cols = select_features(
                df_subs,
                target_sub,
                self.featsel_runs,
                None,
                self.problem_type,
                self.n_jobs,
                self.verbose,
            )
            # if no features were selected, take the original features
            if not good_cols:
                good_cols = list(df.columns)
        else:
            logging.warning(f"[AutoFeat] Unknown problem_type {self.problem_type} - not performing feature selection.")
            good_cols = df_subs.columns
        # filter out those columns that were original features or generated otherwise
        self.new_feat_cols_ = [c for c in good_cols if c not in list(df.columns)]
        self.good_cols_ = good_cols
        # re-generate all good feature again; for all data points this time
        self.feature_functions_ = {}  # type: ignore
        df = self._generate_features(df, self.new_feat_cols_)
        # to prevent an error because sometimes the column names are numpy.str_ instead of normal str
        df.columns = [str(c) for c in df.columns]
        # filter out unnecessary junk from self.feature_formulas_
        self.feature_formulas_ = {f: self.feature_formulas_[f] for f in self.new_feat_cols_ + self.feateng_cols_}
        self.feature_functions_ = {f: self.feature_functions_[f] for f in self.new_feat_cols_}
        self.all_columns_ = list(df.columns)
        # train final prediction model on all selected features
        if self.verbose:
            # final dataframe contains original columns and good additional columns
            logging.info(
                f"[AutoFeat] Final dataframe with {len(df.columns)} feature columns ({len(df.columns) - len(self.original_columns_)} new).",
            )

        # train final prediction model
        if self.problem_type == "regression":
            model = lm.LassoLarsCV(cv=5)
        elif self.problem_type == "classification":
            model = lm.LogisticRegressionCV(cv=5, class_weight="balanced")
        else:
            logging.warning(f"[AutoFeat] Unknown problem_type {self.problem_type} - not fitting a prediction model.")
            model = None
        if model is not None:
            if self.verbose:
                logging.info(f"[AutoFeat] Training final {self.problem_type} model.")
            X = df[self.good_cols_].to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, target)
            self.prediction_model_ = model
            # sklearn requires a "classes_" attribute
            if self.problem_type == "classification":
                self.classes_ = model.classes_
            if self.verbose:
                # for classification, model.coefs_ is n_classes x n_features, but we need n_features
                coefs = model.coef_ if self.problem_type == "regression" else np.max(np.abs(model.coef_), axis=0)
                weights = dict(zip(self.good_cols_, coefs))
                logging.info("[AutoFeat] Trained model: largest coefficients:")
                logging.info(model.intercept_)
                for c in sorted(weights, key=lambda x: abs(weights[x]), reverse=True):
                    if abs(weights[c]) < 1e-5:
                        break
                    logging.info(f"{weights[c]:.6f} * {c}")
                logging.info(f"[AutoFeat] Final score: {model.score(X, target):.4f}")
        if self.always_return_numpy:
            return df.to_numpy()
        return df

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        if self.verbose:
            logging.warning("[AutoFeat] This just calls fit_transform() but does not return the transformed dataframe.")
            logging.info("[AutoFeat] It is much more efficient to call fit_transform() instead of fit() and transform()!")
        _ = self.fit_transform(X, y)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)

        Returns:
            - new_df: new pandas dataframe with all the original features (except categorical features transformed
                      into multiple 0/1 columns) and the most promising engineered features. This df can then be
                      used to train your final model.
        """
        check_is_fitted(self, ["feature_formulas_"])
        # store column names as they'll be lost in the other check
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
        # check input variables
        X = check_array(X, force_all_finite="allow-nan", dtype=None)
        if not cols:
            cols = [f"x{i:03}" for i in range(X.shape[1])]
        if not cols == self.original_columns_:
            raise ValueError("[AutoFeat] Not the same features as when calling fit.")
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # possibly convert categorical columns
        df = self._transform_categorical_cols(df)
        # possibly apply pi-theorem
        df = self._apply_pi_theorem(df)
        # generate engineered features
        df = self._generate_features(df, self.new_feat_cols_)
        # to prevent an error because sometimes the column names are numpy.str_ instead of normal str
        df.columns = [str(c) for c in df.columns]
        if self.always_return_numpy:
            return df.to_numpy()
        return df

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)

        Returns:
            - y_pred: predicted targets returned by prediction_model.predict()
        """
        check_is_fitted(self, ["prediction_model_"])
        df = self._X2df(X)
        return self.prediction_model_.predict(df[self.good_cols_].to_numpy())

    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
            - y: pandas dataframe or numpy array with the targets for all n_datapoints
        Returns:
            - R^2/Accuracy returned by prediction_model.score()
        """
        check_is_fitted(self, ["prediction_model_"])
        # store column names as they'll be lost in the other check
        cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
        # check input variables
        X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression", dtype=None)
        if not cols:
            cols = [f"x{i:03}" for i in range(X.shape[1])]
        # transform X into a dataframe (again)
        df = pd.DataFrame(X, columns=cols)
        # do we need to call transform?
        if list(df.columns) != self.all_columns_:
            temp = self.always_return_numpy
            self.always_return_numpy = False
            df = self.transform(df)
            self.always_return_numpy = temp
        return self.prediction_model_.score(df[self.good_cols_].to_numpy(), target)

    @property
    def n_features_in_(self):
        return len(self.feateng_cols_)


class AutoFeatRegressor(AutoFeatModel, BaseEstimator, RegressorMixin):
    """Short-cut initialization for AutoFeatModel with problem_type: regression"""

    def __init__(
        self,
        categorical_cols: list | None = None,
        feateng_cols: list | None = None,
        units: dict | None = None,
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        max_gb: int | None = None,
        transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
        apply_pi_theorem: bool = True,
        always_return_numpy: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        super().__init__(
            "regression",
            categorical_cols,
            feateng_cols,
            units,
            feateng_steps,
            featsel_runs,
            max_gb,
            transformations,
            apply_pi_theorem,
            always_return_numpy,
            n_jobs,
            verbose,
        )


class AutoFeatClassifier(AutoFeatModel, BaseEstimator, ClassifierMixin):
    """Short-cut initialization for AutoFeatModel with problem_type: classification"""

    def __init__(
        self,
        categorical_cols: list | None = None,
        feateng_cols: list | None = None,
        units: dict | None = None,
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        max_gb: int | None = None,
        transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
        apply_pi_theorem: bool = True,
        always_return_numpy: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        super().__init__(
            "classification",
            categorical_cols,
            feateng_cols,
            units,
            feateng_steps,
            featsel_runs,
            max_gb,
            transformations,
            apply_pi_theorem,
            always_return_numpy,
            n_jobs,
            verbose,
        )

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Inputs:
            - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)

        Returns:
            - y_pred: predicted targets probabilities returned by prediction_model.predict_proba()
        """
        check_is_fitted(self, ["prediction_model_"])
        df = self._X2df(X)
        return self.prediction_model_.predict_proba(df[self.good_cols_].to_numpy())
