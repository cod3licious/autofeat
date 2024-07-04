# Changelog


### 2.1.3 (2024-07-04)

- minor style and type fixes
- improved package structure
- added changelog & docs


### 2.1.2 (2023-07-28)

- converted most print statements to logging outputs
- moved tests and make pytest compatible
- more qa and style fixes (using ruff)


### 2.1.1 (2023-06-25)

- fixed annotations for backwards compatibility


### 2.1.0 (2023-05-14)

- added `predict_proba` functionality for classifier (by @mglowacki100)
- formatting fixes (using black)
- added type hints


### 2.0.10 (2021-10-28)

- fixed issue #29 (by @stephanos-stephani)


### 2.0.9 (2021-06-12)

- speed up correlation computation; fixes issue #28


### 2.0.8 (2021-06-03)

- use numba jit for feature generation (by @jeethu)


### 2.0.7 (2021-06-02)

- use numba for standardization (by @jeethu)


### 2.0.5 (2021-01-16)

- fixed TypeError while running tests with scikit-learn 0.24.0 (by @jeethu)
- minor efficiency improvements in apply_transformations (by @jeethu)
- use numba to accelerate feateng (by @jeethu)


### 2.0.4 (2020-11-30)

- update sympy call to work with new version


### 2.0.3 (2020-11-11)

- turn scaling off by default
- remove more correlated cols by starting with the features that has the most correlated columns


### 2.0.2 (2020-11-11)

- fixed typo


### 2.0.1 (2020-11-11)

- use correlation threshold in autofeat light as parameter


### 2.0.0 (2020-11-07)

- added `AutoFeatLight` model for simple feature selection (removing zero variance and redundant features), engineering (product and ratio of original features) and power transform to make features more normally distributed


### 1.1.3 (2020-07-21)

- categorical columns can contain strings now


### 1.1.2 (2020-02-28)

- don't generate addition/subtr features at the highest level, i.e., if they would just be removed anyways


### 1.1.1 (2020-02-25)

- use LassoLarsCV instead of RidgeCV as final regression model
- minor tweaks to feature selection to avoid longer formulas


### 1.1.0 (2020-02-24)

- include categorical columns for feateng by default
- add correlation filtering back into feat selection


### 1.0.0 (2020-02-24)

- changed autofeat model to differentiate between regression and classification tasks, adding the `AutoFeatRegressor` and `AutoFeatClassifier` classes
- simplified feature selection process


### 0.2.5 (2019-05-12)

- more robust featsel with noise filtering


### 0.2.2 (2019-05-09)

- change default value for `feateng_steps` to 2, in line with results on realworld datasets


### 0.2.1 (2019-05-09)

- make feature selection less prone to overfitting


### 0.2.0 (2019-05-02)

- add `FeatureSelector` class to use feature selection separately
- make feature selection more robust and move into featsel
- make the models more sklearn like and test with sklearn estimator tests
- replace sympy's ufuncify with lambdify
- better logs
- use immutable default arguments
- make pi theorem optional
- handle nans in transform


### 0.1.1 (2019-01-23)

- updated documentation


### 0.1.0 (2019-01-22)

- initial release with regression model
