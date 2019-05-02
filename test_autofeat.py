import numpy as np
import pandas as pd

from autofeat import AutoFeatRegression
from test_estimator import check_estimator_autofeat


def get_random_data(seed=15):
    # generate some toy data
    np.random.seed(seed)
    x1 = np.random.rand(1000)
    x2 = np.random.randn(1000)
    x3 = np.random.rand(1000)
    target = 2 + 15*x1 + 3/(x2 - 1/x3) + 5*(x2 + np.log(x1))**3
    X = np.vstack([x1, x2, x3]).T
    return X, target


def test_do_almost_nothing():
    X, target = get_random_data()
    afreg = AutoFeatRegression(verbose=1, feateng_steps=0, featsel_runs=0)
    df = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target)
    assert list(df.columns) == ["x1", "x2", "x3"], "Only original columns"
    df = afreg.transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]))
    assert list(df.columns) == ["x1", "x2", "x3"], "Only original columns"


def test_regular_X_y():
    # autofeat with numpy arrays
    X, target = get_random_data()
    afreg = AutoFeatRegression(verbose=1)
    df = afreg.fit_transform(X, target)
    assert afreg.score(X, target) == 1., "R^2 should be 1."
    assert afreg.score(df, target) == 1., "R^2 should be 1."
    assert list(df.columns)[:3] == ["x000", "x001", "x002"], "Wrong column names"


def test_regular_df_X_y():
    # autofeat with df without column names
    X, target = get_random_data()
    afreg = AutoFeatRegression(verbose=1)
    df = afreg.fit_transform(pd.DataFrame(X), pd.DataFrame(target))
    # score once with original, once with transformed data
    assert afreg.score(pd.DataFrame(X), target) == 1., "R^2 should be 1."
    assert afreg.score(df, target) == 1., "R^2 should be 1."
    assert list(df.columns)[:3] == ["0", "1", "2"], "Wrong column names"


def test_weird_colnames():
    # autofeat with df with weird column names
    X, target = get_random_data()
    afreg = AutoFeatRegression(verbose=1)
    df = afreg.fit_transform(pd.DataFrame(X, columns=["x 1.1", 2, "x/3"]), pd.DataFrame(target))
    assert afreg.score(pd.DataFrame(X, columns=["x 1.1", 2, "x/3"]), target) == 1., "R^2 should be 1."
    assert list(df.columns)[:3] == ["x 1.1", "2", "x/3"], "Wrong column names"
    # error if the column names aren't the same as before
    try:
        afreg.score(pd.DataFrame(X, columns=["x 11", 2, "x/3"]), target)
    except ValueError:
        pass
    else:
        raise AssertionError("Should throw error on mismatch column names")


def test_nans():
    # nans are ok in transform but not fit or predict (due to sklearn model)
    X, target = get_random_data()
    X[998, 0] = np.nan
    X[999, 1] = np.nan
    afreg = AutoFeatRegression(verbose=1)
    try:
        _ = afreg.fit_transform(pd.DataFrame(X, columns=["x 1.1", 2, "x/3"]), pd.DataFrame(target))
    except ValueError:
        pass
    else:
        raise AssertionError("fit with NaNs should throw an error")
    _ = afreg.fit_transform(pd.DataFrame(X[:900], columns=["x 1.1", 2, "x/3"]), pd.DataFrame(target[:900]))
    try:
        _ = afreg.predict(pd.DataFrame(X[900:], columns=["x 1.1", 2, "x/3"]))
    except ValueError:
        pass
    else:
        raise AssertionError("predict with NaNs should throw an error")
    df = afreg.transform(pd.DataFrame(X, columns=["x 1.1", 2, "x/3"]))
    assert all([pd.isna(df.iloc[998, 0]), pd.isna(df.iloc[999, 1])]), "Original features should be NaNs"
    assert np.sum(np.array(pd.isna(df.iloc[998]), dtype=int)) >= 2, "There should be at least 2 NaNs in row 998"
    assert np.sum(np.array(pd.isna(df.iloc[999]), dtype=int)) >= 2, "There should be at least 3 NaNs in row 999"


def test_feateng_cols():
    X, target = get_random_data()
    afreg = AutoFeatRegression(verbose=1, feateng_cols=["x1", "x3", "x4"])
    try:
        df = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target)
    except ValueError:
        pass
    else:
        raise AssertionError("feateng_cols not in df should throw an error")
    afreg = AutoFeatRegression(verbose=1, feateng_cols=["x1", "x3"])
    df = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target)
    for c in df.columns[3:]:
        assert "x2" not in c, "only feateng_cols should occur in engineered features"


def test_categorical_cols():
    np.random.seed(15)
    x1 = np.random.rand(1000)
    x2 = np.random.randn(1000)
    x3 = np.random.rand(1000)
    x4 = np.array(200*[4] + 300*[5] + 500*[2], dtype=int)
    target = 2 + 15*x1 + 3/(x2 - 1/x3) + 5*(x2 + np.log(x1))**3 + x4
    X = np.vstack([x1, x2, x3, x4]).T
    afreg = AutoFeatRegression(verbose=1, categorical_cols=["x4", "x5"])
    try:
        df = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"]), target)
    except ValueError:
        pass
    else:
        raise AssertionError("categorical_cols not in df should throw an error")
    afreg = AutoFeatRegression(verbose=1, categorical_cols=["x4"])
    df = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"]), target)
    assert list(df.columns)[3:6] == ["x4_2.0", "x4_4.0", "x4_5.0"], "categorical_cols were not transformed correctly"
    assert "x4" not in df.columns, "categorical_cols weren't deleted from df"
    df = afreg.transform(pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"]))
    assert list(df.columns)[3:6] == ["x4_2.0", "x4_4.0", "x4_5.0"], "categorical_cols were not transformed correctly"
    assert "x4" not in df.columns, "categorical_cols weren't deleted from df"
    assert afreg.score(pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"]), target) == 1., "R^2 should be 1."


def test_units():
    np.random.seed(15)
    x1 = np.random.rand(1000)
    x2 = np.random.randn(1000)
    x3 = np.random.rand(1000)
    target = 2 + 15*x1 + 3/(x2 - 1/x3) + 5*(x2 * np.log(x1))**3
    X = np.vstack([x1, x2, x3]).T
    units = {"x2": "m/sec", "x3": "min/mm"}
    afreg = AutoFeatRegression(verbose=1, units=units)
    _ = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target)
    assert afreg.score(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target) == 1., "R^2 should be 1."


if __name__ == '__main__':
    check_estimator_autofeat(AutoFeatRegression)
    test_do_almost_nothing()
    test_regular_X_y()
    test_regular_df_X_y()
    test_weird_colnames()
    test_nans()
    test_feateng_cols()
    test_categorical_cols()
    test_units()
