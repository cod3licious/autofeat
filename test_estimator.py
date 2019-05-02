import warnings
import numpy as np
from scipy import sparse
from sklearn.base import (clone, ClusterMixin,
                          is_classifier, is_regressor,
                          is_outlier_detector)
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import set_random_state
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import SkipTest
from sklearn.exceptions import SkipTestWarning, DataConversionWarning
from sklearn.utils.estimator_checks import (
    pairwise_estimator_convert_X,
    multioutput_estimator_convert_y_2d,
    check_parameters_default_constructible,
    check_no_attributes_set_in_init,
    _yield_non_meta_checks,
    _yield_classifier_checks,
    _yield_regressor_checks,
    _yield_transformer_checks,
    _yield_clustering_checks,
    _yield_outliers_checks,
    check_fit2d_predict1d,
    check_fit2d_1sample,
    check_fit2d_1feature,
    check_fit1d,
    check_get_params_invariance,
    check_set_params,
    check_dict_unchanged,
    check_dont_overwrite_parameters)


def _apply_on_subsets(func, X):
    # apply function on the whole set and on mini batches
    result_full = func(X)
    n_features = X.shape[1]
    result_by_batch = [func(batch.reshape(1, n_features))
                       for batch in X]
    # func can output tuple (e.g. score_samples)
    if type(result_full) == tuple:
        result_full = result_full[0]
        result_by_batch = list(map(lambda x: x[0], result_by_batch))

    if sparse.issparse(result_full):
        result_full = result_full.A
        result_by_batch = [x.A for x in result_by_batch]
    return np.ravel(result_full), np.ravel([np.ravel(b) for b in result_by_batch])


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini bathes or the whole set
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function",
                   "score_samples", "predict_proba"]:

        msg = ("{method} of {name} is not invariant when applied "
               "to a subset.").format(method=method, name=name)
        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_on_subsets(
                getattr(estimator, method), X)
            assert_allclose(result_full, result_by_batch,
                            atol=1e-7, err_msg=msg)


def _yield_all_checks(name, estimator):
    for check in _yield_non_meta_checks(name, estimator):
        yield check
    if is_classifier(estimator):
        for check in _yield_classifier_checks(name, estimator):
            yield check
    if is_regressor(estimator):
        for check in _yield_regressor_checks(name, estimator):
            yield check
    if hasattr(estimator, 'transform'):
        for check in _yield_transformer_checks(name, estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(name, estimator):
            yield check
    if is_outlier_detector(estimator):
        for check in _yield_outliers_checks(name, estimator):
            yield check
    yield check_fit2d_predict1d
    yield check_methods_subset_invariance
    yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_fit1d
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters


def check_estimator(Estimator):
    """Check if estimator adheres to scikit-learn conventions.
    This estimator will run an extensive test-suite for input validation,
    shapes, etc.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.
    This test can be applied to classes or instances.
    Classes currently have some additional tests that related to construction,
    while passing instances allows the testing of multiple options.
    Parameters
    ----------
    estimator : estimator object or class
        Estimator to check. Estimator is a class object or instance.
    """
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    if isinstance(Estimator, type):
        # got a class
        name = Estimator.__name__
        estimator = Estimator()
        check_parameters_default_constructible(name, Estimator)
        check_no_attributes_set_in_init(name, estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__

    for check in _yield_all_checks(name, estimator):
        if hasattr(check, "__name__"):
            if check.__name__ == "check_estimators_nan_inf":
                # we allow NaNs in transform...
                continue
            print("##", check.__name__)
        try:
            check(name, estimator)
        except SkipTest as exception:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(str(exception), SkipTestWarning)


def check_estimator_autofeat(Estimator):
    # usually, this would be
    # from sklearn.utils.estimator_checks import check_estimator
    # but first this issue needs to be resolved:
    # https://github.com/pandas-dev/pandas/issues/26247
    # check class
    name = Estimator.__name__
    estimator = Estimator()
    check_parameters_default_constructible(name, Estimator)
    check_no_attributes_set_in_init(name, estimator)
    # check with fewer feateng steps and featsel runs to speed things up
    check_estimator(Estimator(feateng_steps=1, featsel_runs=1))
