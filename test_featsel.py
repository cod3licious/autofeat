import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from autofeat import FeatureSelector


def get_random_data(seed=15):
    # generate some toy data
    np.random.seed(seed)
    x1 = np.random.rand(1000)
    x2 = np.random.randn(1000)
    x3 = np.random.rand(1000)
    x4 = np.random.randn(1000)
    x5 = np.random.rand(1000)
    target = 2 + 15*x1 + 3/(x2 - 1/x3) + 5*(x2 + np.log(x1))**3
    X = np.vstack([x1, x2, x3, x4, x5, 1/(x2 - 1/x3), (x2 + np.log(x1))**3]).T
    return X, target


def test_regular_X_y():
    # featsel with numpy arrays
    X, target = get_random_data()
    fsel = FeatureSelector(verbose=0)
    new_X = fsel.fit_transform(X, target)
    assert isinstance(new_X, np.ndarray)
    assert new_X.shape[1] == 3, "Wrong number of features selected"


def test_regular_df_X_y():
    # featsel with df without column names
    X, target = get_random_data()
    fsel = FeatureSelector(verbose=0)
    new_X = fsel.fit_transform(pd.DataFrame(X), pd.DataFrame(target))
    assert isinstance(new_X, pd.DataFrame)
    assert set(new_X.columns) == set([0, 5, 6]), "Wrong features selected (%r)" % new_X.columns


def test_df_X_y():
    # featsel with df with column names
    X, target = get_random_data()
    fsel = FeatureSelector(verbose=0)
    new_X = fsel.fit_transform(pd.DataFrame(X, columns=[1, 2, "3", "x4", "x5", "eng6", "eng7"]), target)
    assert isinstance(new_X, pd.DataFrame)
    assert set(new_X.columns) == set(["1", "eng6", "eng7"]), "Wrong features selected (%r)" % new_X.columns


def test_keep():
    # featsel with df with column names
    X, target = get_random_data()
    fsel = FeatureSelector(verbose=0, keep=[2, "x5"])
    new_X = fsel.fit_transform(pd.DataFrame(X, columns=[1, 2, "3", "x4", "x5", "eng6", "eng7"]), target)
    assert isinstance(new_X, pd.DataFrame)
    assert set(new_X.columns) == set(["1", "eng6", "eng7", "2", "x5"]), "Wrong features selected (%r)" % new_X.columns


def test_nans():
    # featsel with df without column names
    X, target = get_random_data()
    X[998, 0] = np.nan
    X[999, 1] = np.nan
    fsel = FeatureSelector(verbose=0)
    try:
        _ = fsel.fit(pd.DataFrame(X), target)
    except ValueError:
        pass
    else:
        raise AssertionError("fit with NaNs should throw an error")
    _ = fsel.fit_transform(pd.DataFrame(X[:900]), target[:900])
    df = fsel.transform(pd.DataFrame(X))
    assert pd.isna(df[0].iloc[998]), "The first feature should be NaN"
    assert np.sum(pd.isna(df).to_numpy(dtype=int)) == 1, "only 1 place should be NaN"
    assert set(df.columns) == set([0, 5, 6]), "Wrong features selected (%r)" % df.columns


if __name__ == '__main__':
    print("## Running sklearn tests")
    # we allow for nan in transform
    successful_tests = set(["check_estimators_nan_inf"])
    for estimator, check in check_estimator(FeatureSelector(featsel_runs=1), generate_only=True):
        if check.func.__name__ not in successful_tests:
            print(check.func.__name__)
            successful_tests.add(check.func.__name__)
            check(estimator)
    # additionally check the class, but don't run all the other tests
    for estimator, check in check_estimator(FeatureSelector(), generate_only=True):
        if check.func.__name__ not in successful_tests:
            print(check.func.__name__)
            successful_tests.add(check.func.__name__)
            check(estimator)

    print("## Running custom tests")
    test_regular_X_y()
    test_regular_df_X_y()
    test_df_X_y()
    test_keep()
    test_nans()
    print("## Looks like all tests were successful :)")
