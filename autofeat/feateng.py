# -*- coding: utf-8 -*-
# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import str
import re
import operator as op
from functools import reduce
from itertools import combinations, product
import numpy as np
import pandas as pd
import sympy
from sympy.utilities.lambdify import lambdify
import pint


def colnames2symbols(c, i=0):
    # take a messy column name and transform it to something sympy can handle
    # worst case: i is the number of the features
    # has to be a string
    c = str(c)
    # should not contain non-alphanumeric characters
    c = re.sub(r"\W+", "", c)
    if not c:
        c = "x%03i" % i
    elif c[0].isdigit():
        c = "x" + c
    return c


def ncr(n, r):
    # compute number of combinations for n chose r
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def n_cols_generated(n_features, max_steps, n_transformations=7, n_combinations=4):
    """
    computes the upper bound of how many features will be generated based on n_features to start with
    and max_steps feateng steps.
    """
    # n_transformations is 1-len(func_transform) because either abs() or sqrt and log will be applied
    n_transformations -= 1
    original_cols = n_features
    new_cols = 0
    new_new_cols = 0
    # count additions at the highest level
    n_additions = 0
    steps = 1
    if steps <= max_steps:
        # Step 1: apply transformations to original features
        original_cols += n_features * n_transformations
        # n_additions += n_features * 2  # only if 1+ or 1- is in transformations!
        steps += 1
    if steps <= max_steps:
        # Step 2: first combination of features
        new_cols = n_combinations * (ncr(original_cols, 2))
        n_additions += 3 * new_cols // 4
        steps += 1
    while steps <= max_steps:
        # apply transformations on these new features
        # n_additions += new_cols * 2
        new_cols += new_cols * n_transformations
        steps += 1
        # get combinations of old and new features
        if steps <= max_steps:
            new_new_cols = n_combinations * (original_cols * new_cols)
            n_additions += 3 * new_new_cols // 4
            steps += 1
        # and combinations of new features within themselves
        if steps <= max_steps:
            n = n_combinations * (ncr(new_cols, 2))
            new_new_cols += n
            n_additions += 3 * n // 4
            steps += 1
            # update old and new features and repeat
            original_cols += new_cols
            new_cols = new_new_cols
            new_new_cols = 0
    # finally, apply transformation on the last new features
    if steps <= max_steps:
        # n_additions += new_cols * 2
        new_cols += new_cols * n_transformations
    return original_cols + new_cols + new_new_cols - n_additions


def engineer_features(
    df_org,
    start_features=None,
    units=None,
    max_steps=3,
    transformations=("exp", "log", "abs", "sqrt", "^2", "^3", "1/"),
    verbose=0,
):
    """
    Given a DataFrame with original features, perform the feature engineering routine for max_steps.
    It starts with a transformation of the original features (applying log, ^2, sqrt, etc.),
    then in the next step, the features are combined (x+y, x*y, ...), and in further steps, the resulting
    features are again transformed and combinations of the resulting features are computed.

    Inputs:
        - df_org: pandas DataFrame with original features in columns
        - start_features: list with column names for df_org with features that should be considered for expansion
                          (default: None --> all columns)
        - units: a dict with {column_name: pint.Quantity}: some operations like x+y can only be performed if the
                 features have comparable units (default None: all combinations are allowed)
                 careful: will be modified in place!
        - max_steps: how many feature engineering steps should be performed. Default is 3, this produces:
            Step 1: transformation of original features
            Step 2: first combination of features
            Step 3: transformation of new features
            (Step 4: combination of old and new features)
            --> with 3 original features, after 4 steps you will already end up with around 200k features!
        - transformations: list of transformations that should be applied; possible elements:
                           "exp", "log", "abs", "sqrt", "^2", "^3", "1/", "1+", "1-", "sin", "cos", "exp-", "2^"
                           (first 7, i.e., up to 1/, are applied by default)
        - verbose: verbosity level (int; default: 0)
    Returns:
        - df: new DataFrame with all features in columns
        - feature_pool: dict with {col: sympy formula} formulas to generate each feature
    """
    # initialize the feature pool with columns from the dataframe
    if not start_features:
        start_features = df_org.columns
    else:
        for c in start_features:
            if c not in df_org.columns:
                raise ValueError("[feateng] start feature %r not in df_org.columns" % c)
    feature_pool = {c: sympy.symbols(colnames2symbols(c, i), real=True) for i, c in enumerate(start_features)}
    if max_steps < 1:
        if verbose:
            print("[feateng] Warning: no features generated for max_steps < 1.")
        return df_org, feature_pool
    # get a copy of the dataframe - this is where all the features will be added
    df = pd.DataFrame(df_org.copy(), dtype=np.float32)

    def apply_tranformations(features_list):
        # feature transformations
        func_transform = {
            "exp": lambda x: sympy.exp(x),
            "exp-": lambda x: sympy.exp(-x),
            "log": lambda x: sympy.log(x),
            "abs": lambda x: sympy.Abs(x),
            "sqrt": lambda x: sympy.sqrt(x),
            "sin": lambda x: sympy.sin(x),
            "cos": lambda x: sympy.cos(x),
            "2^": lambda x: 2**x,
            "^2": lambda x: x**2,
            "^3": lambda x: x**3,
            "1+": lambda x: 1 + x,
            "1-": lambda x: 1 - x,
            "1/": lambda x: 1 / x
        }
        func_transform_units = {
            "exp": lambda x: np.exp(x),
            "exp-": lambda x: np.exp(-x),
            "log": lambda x: np.log(x),
            "abs": lambda x: np.abs(x),
            "sqrt": lambda x: np.sqrt(x),
            "sin": lambda x: np.sin(x),
            "cos": lambda x: np.cos(x),
            "2^": lambda x: np.exp(x),
            "^2": lambda x: x**2,
            "^3": lambda x: x**3,
            "1+": lambda x: 1 + x,
            "1-": lambda x: 1 - x,
            "1/": lambda x: 1 / x
        }
        # conditions on the original features that have to be met to apply the transformation
        func_transform_cond = {
            "exp": lambda x: np.all(x < 10),
            "exp-": lambda x: np.all(-x < 10),
            "log": lambda x: np.all(x > 0),
            "abs": lambda x: np.any(x < 0),
            "sqrt": lambda x: np.all(x >= 0),
            "sin": lambda x: True,
            "cos": lambda x: True,
            "2^": lambda x: np.all(x < 50),
            "^2": lambda x: np.all(np.abs(x) < 1000000),
            "^3": lambda x: np.all(np.abs(x) < 10000),
            "1+": lambda x: True,
            "1-": lambda x: True,
            "1/": lambda x: np.all(x != 0)
        }
        # apply transformations to the features in the given features list
        # modifies global variables df and feature_pool!
        nonlocal df, feature_pool, units
        # returns a list of new features that were generated
        new_features = []
        # store all new features in a preallocated numpy array before adding it to the dataframe
        feat_array = np.zeros((df.shape[0], len(features_list) * len(transformations)), dtype=np.float32)
        for i, feat in enumerate(features_list):
            if verbose and not i % 100:
                print("[feateng] %15i/%15i features transformed" % (i, len(features_list)), end="\r")
            for ft in transformations:
                # check if transformation is valid for particular feature (i.e. given actual numerical values)
                if func_transform_cond[ft](df[feat]):
                    # get the expression (based on the primary features)
                    expr = func_transform[ft](feature_pool[feat])
                    expr_name = str(expr)
                    # we're simplifying expressions, so we might already have that one
                    if expr_name not in feature_pool:
                        # if we're given units, check if the operation is legal
                        if units:
                            try:
                                units[expr_name] = func_transform_units[ft](units[feat])
                                units[expr_name].__dict__["_magnitude"] = 1.
                            except (pint.DimensionalityError, pint.OffsetUnitCalculusError):
                                continue
                        feature_pool[expr_name] = expr
                        # create temporary variable expression and apply it to precomputed feature
                        t = sympy.symbols("t")
                        expr_temp = func_transform[ft](t)
                        f = lambdify(t, expr_temp)
                        new_feat = np.array(f(df[feat].to_numpy()), dtype=np.float32)
                        if np.isfinite(new_feat).all():
                            feat_array[:, len(new_features)] = new_feat
                            new_features.append(expr_name)
        if verbose:
            print("[feateng] Generated %i transformed features from %i original features - done." % (len(new_features), len(features_list)))
        df = df.join(pd.DataFrame(feat_array[:, :len(new_features)], columns=new_features, index=df.index, dtype=np.float32))
        return new_features

    def get_feature_combinations(feature_tuples):
        # new features as combinations of two other features
        func_combinations = {
            "x+y": lambda x, y: x + y,
            "x*y": lambda x, y: x * y,
            "x-y": lambda x, y: x - y,
            "y-x": lambda x, y: y - x
        }
        # get all feature combinations for the given feature tuples
        # modifies global variables df and feature_pool!
        nonlocal df, feature_pool, units
        # returns a list of new features that were generated
        new_features = []
        # store all new features in a preallocated numpy array before adding it to the dataframe
        feat_array = np.zeros((df.shape[0], len(feature_tuples) * len(func_combinations)), dtype=np.float32)
        for i, (feat1, feat2) in enumerate(feature_tuples):
            if verbose and not i % 100:
                print("[feateng] %15i/%15i feature tuples combined" % (i, len(feature_tuples)), end="\r")
            for fc in func_combinations:
                expr = func_combinations[fc](feature_pool[feat1], feature_pool[feat2])
                expr_name = str(expr)
                if expr_name not in feature_pool:
                    # if we're given units, check if the operation is legal
                    if units:
                        try:
                            units[expr_name] = func_combinations[fc](units[feat1], units[feat2])
                            units[expr_name].__dict__["_magnitude"] = 1.
                        except (pint.DimensionalityError, pint.OffsetUnitCalculusError):
                            continue
                    feature_pool[expr_name] = expr
                    # create temporary variable expression to apply it to precomputed features
                    s, t = sympy.symbols("s t")
                    expr_temp = func_combinations[fc](s, t)
                    f = lambdify((s, t), expr_temp)
                    new_feat = np.array(f(df[feat1].to_numpy(), df[feat2].to_numpy()), dtype=np.float32)
                    if np.isfinite(new_feat).all():
                        feat_array[:, len(new_features)] = new_feat
                        new_features.append(expr_name)
        if verbose:
            print("[feateng] Generated %i feature combinations from %i original feature tuples - done." % (len(new_features), len(feature_tuples)))
        df = df.join(pd.DataFrame(feat_array[:, :len(new_features)], columns=new_features, index=df.index, dtype=np.float32))
        return new_features

    # get transformations of initial features
    steps = 1
    if verbose:
        print("[feateng] Step 1: transformation of original features")
    original_features = list(feature_pool.keys())
    original_features.extend(apply_tranformations(original_features))
    steps += 1
    # get combinations of first feature set
    if steps <= max_steps:
        if verbose:
            print("[feateng] Step 2: first combination of features")
        new_features = get_feature_combinations(list(combinations(original_features, 2)))
        steps += 1
    while steps <= max_steps:
        # apply transformations on these new features
        if verbose:
            print("[feateng] Step %i: transformation of new features" % steps)
        new_features.extend(apply_tranformations(new_features))
        steps += 1
        # get combinations of old and new features
        if steps <= max_steps:
            if verbose:
                print("[feateng] Step %i: combination of old and new features" % steps)
            new_new_features = get_feature_combinations(list(product(original_features, new_features)))
            steps += 1
        # and combinations of new features within themselves
        if steps <= max_steps:
            if verbose:
                print("[feateng] Step %i: combination of new features" % steps)
            new_new_features.extend(get_feature_combinations(list(combinations(new_features, 2))))
            steps += 1
            # update old and new features and repeat
            original_features.extend(new_features)
            new_features = new_new_features
    # finally, apply transformation on the last new features
    if steps <= max_steps:
        if verbose:
            print("[feateng] Step %i: transformation of last new features" % steps)
        new_features.extend(apply_tranformations(new_features))

    # sort out all features that are just additions on the highest level
    cols = [c for c in list(df.columns) if not (c in feature_pool and feature_pool[c].func == sympy.add.Add)]
    if verbose:
        print("[feateng] Generated a total of %i additional features" % (len(cols) - len(start_features)))
    return df[cols], feature_pool
