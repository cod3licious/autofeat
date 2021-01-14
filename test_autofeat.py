import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from autofeat import AutoFeatRegressor, AutoFeatClassifier


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
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=0, featsel_runs=0)
    df = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target)
    assert list(df.columns) == ["x1", "x2", "x3"], "Only original columns"
    df = afreg.transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]))
    assert list(df.columns) == ["x1", "x2", "x3"], "Only original columns"


def test_regular_X_y():
    # autofeat with numpy arrays
    X, target = get_random_data()
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)
    df = afreg.fit_transform(X, target)
    assert afreg.score(X, target) >= 0.999, "R^2 should be 1."
    assert afreg.score(df, target) >= 0.999, "R^2 should be 1."
    assert list(df.columns)[:3] == ["x000", "x001", "x002"], "Wrong column names"


def test_regular_df_X_y():
    # autofeat with df without column names
    X, target = get_random_data()
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)
    df = afreg.fit_transform(pd.DataFrame(X), pd.DataFrame(target))
    # score once with original, once with transformed data
    assert afreg.score(pd.DataFrame(X), target) >= 0.999, "R^2 should be 1."
    assert afreg.score(df, target) >= 0.999, "R^2 should be 1."
    assert list(df.columns)[:3] == ["0", "1", "2"], "Wrong column names"


def test_weird_colnames():
    # autofeat with df with weird column names
    X, target = get_random_data()
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)
    df = afreg.fit_transform(pd.DataFrame(X, columns=["x 1.1", 2, "x/3"]), pd.DataFrame(target))
    assert afreg.score(pd.DataFrame(X, columns=["x 1.1", 2, "x/3"]), target) >= 0.999, "R^2 should be 1."
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
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)
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
    afreg = AutoFeatRegressor(verbose=1, feateng_cols=["x1", "x3", "x4"], feateng_steps=3)
    try:
        df = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target)
    except ValueError:
        pass
    else:
        raise AssertionError("feateng_cols not in df should throw an error")
    afreg = AutoFeatRegressor(verbose=1, feateng_cols=["x1", "x3"], feateng_steps=3)
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
    X = pd.DataFrame(np.vstack([x1, x2, x3, x4]).T, columns=["x1", "x2", "x3", "x4"])
    X["x4"] = np.array(200*[4] + 300*["hello"] + 500*[2])  # categories can be weird strings
    afreg = AutoFeatRegressor(verbose=1, categorical_cols=["x4", "x5"], feateng_steps=3)
    try:
        df = afreg.fit_transform(X, target)
    except ValueError:
        pass
    else:
        raise AssertionError("categorical_cols not in df should throw an error")
    afreg = AutoFeatRegressor(verbose=1, categorical_cols=["x4"], feateng_steps=3)
    df = afreg.fit_transform(X, target)
    assert list(df.columns)[3:6] == ["cat_x4_'2'", "cat_x4_'4'", "cat_x4_'hello'"], "categorical_cols were not transformed correctly"
    assert "x4" not in df.columns, "categorical_cols weren't deleted from df"
    df = afreg.transform(X)
    assert list(df.columns)[3:6] == ["cat_x4_'2'", "cat_x4_'4'", "cat_x4_'hello'"], "categorical_cols were not transformed correctly"
    assert "x4" not in df.columns, "categorical_cols weren't deleted from df"
    assert afreg.score(X, target) >= 0.999, "R^2 should be 1."


def test_units():
    np.random.seed(15)
    x1 = np.random.rand(1000)
    x2 = np.random.randn(1000)
    x3 = np.random.rand(1000)
    target = 2 + 15*x1 + 3/(x2 - 1/x3) + 5*(x2 * np.log(x1))**3
    X = np.vstack([x1, x2, x3]).T
    units = {"x2": "m/sec", "x3": "min/mm"}
    afreg = AutoFeatRegressor(verbose=1, units=units, feateng_steps=3)
    _ = afreg.fit_transform(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target)
    assert afreg.score(pd.DataFrame(X, columns=["x1", "x2", "x3"]), target) >= 0.999, "R^2 should be 1."


def test_classification():
    # autofeat with numpy arrays but as classification
    X, target = get_random_data()
    target = np.array(target > target.mean(), dtype=int)
    afreg = AutoFeatClassifier(verbose=1, feateng_steps=3)
    df = afreg.fit_transform(X, target)
    assert afreg.score(X, target) >= 0.9999, "Accuracy should be 1."
    assert afreg.score(df, target) >= 0.9999, "Accuracy should be 1."
    assert list(df.columns)[:3] == ["x000", "x001", "x002"], "Wrong column names"


if __name__ == '__main__':
    print("## Running sklearn Regressor tests")
    # we allow for nan in transform
    successful_tests = set(["check_estimators_nan_inf"])
    for estimator, check in check_estimator(AutoFeatRegressor(feateng_steps=1, featsel_runs=1, always_return_numpy=True), generate_only=True):
        if check.func.__name__ not in successful_tests:
            print(check.func.__name__)
            successful_tests.add(check.func.__name__)
            check(estimator)
    # additionally check the class, but don't run all the other tests
    for estimator, check in check_estimator(AutoFeatRegressor(), generate_only=True):
        if check.func.__name__ not in successful_tests:
            print(check.func.__name__)
            successful_tests.add(check.func.__name__)
            check(estimator)

    print("## Running sklearn Classifier tests")
    # we allow for nan in transform
    successful_tests = set(["check_estimators_nan_inf"])
    for estimator, check in check_estimator(AutoFeatClassifier(feateng_steps=1, featsel_runs=1, always_return_numpy=True), generate_only=True):
        if check.func.__name__ not in successful_tests:
            print(check.func.__name__)
            successful_tests.add(check.func.__name__)
            check(estimator)
    # additionally check the class, but don't run all the other tests
    for estimator, check in check_estimator(AutoFeatClassifier(), generate_only=True):
        if check.func.__name__ not in successful_tests:
            print(check.func.__name__)
            successful_tests.add(check.func.__name__)
            check(estimator)

    print("## Running custom tests")
    print("# test_do_almost_nothing")
    test_do_almost_nothing()
    print("# test_regular_X_y")
    test_regular_X_y()
    print("# test_regular_df_X_y")
    test_regular_df_X_y()
    print("# test_weird_colnames")
    test_weird_colnames()
    print("# test_nans")
    test_nans()
    print("# test_feateng_cols")
    test_feateng_cols()
    print("# test_categorical_cols")
    test_categorical_cols()
    print("# test_units")
    test_units()
    print("## Looks like all tests were successful :)")
