import numpy as np
import pandas as pd

from test_estimator import check_estimator
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


def test_few_runs():
    # featsel with numpy arrays
    X, target = get_random_data()
    fsel = FeatureSelector(verbose=0, featsel_runs=0)
    new_X = fsel.fit_transform(X, target)
    assert new_X.shape[1] == 3, "Wrong number of features selected"
    fsel = FeatureSelector(verbose=0, featsel_runs=1)
    new_X = fsel.fit_transform(X, target)
    assert new_X.shape[1] == 3, "Wrong number of features selected"


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
    assert set(new_X.columns) == set([0, 5, 6]), "Wrong features selected"


def test_df_X_y():
    # featsel with df with column names
    X, target = get_random_data()
    fsel = FeatureSelector(verbose=0)
    new_X = fsel.fit_transform(pd.DataFrame(X, columns=[1, 2, "3", "x4", "x5", "eng6", "eng7"]), pd.DataFrame(target))
    assert isinstance(new_X, pd.DataFrame)
    assert set(new_X.columns) == set([1, "eng6", "eng7"]), "Wrong features selected"


def test_nans():
    # featsel with df without column names
    X, target = get_random_data()
    X[98, 0] = np.nan
    X[99, 1] = np.nan
    fsel = FeatureSelector(verbose=0)
    try:
        _ = fsel.fit(pd.DataFrame(X), pd.DataFrame(target))
    except ValueError:
        pass
    else:
        raise AssertionError("fit with NaNs should throw an error")
    _ = fsel.fit_transform(pd.DataFrame(X[:90]), pd.DataFrame(target[:90]))
    df = fsel.transform(pd.DataFrame(X))
    assert pd.isna(df[0].iloc[98]), "The first feature should be NaN"
    assert np.sum(pd.isna(df).to_numpy(dtype=int)) == 1, "only 1 place should be NaN"


if __name__ == '__main__':
    test_regular_X_y()
    test_regular_df_X_y()
    test_df_X_y()
    test_nans()
    check_estimator(FeatureSelector)
