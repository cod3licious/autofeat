# Getting Started

This library contains the `AutoFeatRegressor` and `AutoFeatClassifier` models with a similar interface as `scikit-learn` models:

- `fit()` function to fit the model parameters
- `predict()` function to predict the target variable given the input
- `predict_proba()` function to predict probabilities of the target variable given the input (classifier only)
- `score()` function to calculate the goodness of the fit (R^2/accuracy)
- `fit_transform()` and `transform()` functions, which extend the given data by the additional features that were engineered and selected by the model

When calling the `fit()` function, internally the `fit_transform()` function will be called, so if you're planing to call `transform()` on the same data anyways, just call `fit_transform()` right away. `transform()` is mostly useful if you've split your data into training and test data and did not call `fit_transform()` on your whole dataset. The `predict()` and `score()` functions can either be given data in the format of the original dataframe that was used when calling `fit()`/`fit_transform()` or they can be given an already transformed dataframe.

In addition, only the feature selection part is also available in the `FeatureSelector` model.

Furthermore (as of version 2.0.0), minimal feature selection (removing zero variance and redundant features), engineering (simple product and ratio of features), and scaling (power transform to make features more normally distributed) is also available in the `AutoFeatLight` model.

The `AutoFeatRegressor`, `AutoFeatClassifier`, and `FeatureSelector` models need to be **fit on data without NaNs**, as they internally call the sklearn `LassoLarsCV` model, which can not handle NaNs. When calling `transform()`, NaNs (but not `np.inf`) are okay.

The [autofeat examples notebook](https://github.com/cod3licious/autofeat/blob/main/notebooks/autofeat_examples.ipynb) contains a simple usage example - try it out! :) Additional examples can be found in the autofeat benchmark notebooks for [regression](https://github.com/cod3licious/autofeat/blob/main/notebooks/autofeat_benchmark_regression.ipynb) (which also contains the code to reproduce the results from the paper mentioned below) and [classification](https://github.com/cod3licious/autofeat/blob/main/notebooks/autofeat_benchmark_classification.ipynb), as well as the testing scripts.

Please keep in mind that since the `AutoFeatRegressor` and `AutoFeatClassifier` models can generate very complex features, they might **overfit on noise** in the dataset, i.e., find some new features that lead to good prediction on the training set but result in a poor performance on new test samples. While this usually only happens for datasets with very few samples, we suggest you carefully inspect the features found by `autofeat` and use those that make sense to you to train your own models.

Depending on the number of `feateng_steps` (default 2) and the number of input features, `autofeat` can generate a very huge feature matrix (before selecting the most appropriate features from this large feature pool). By specifying in `feateng_cols` those columns that you expect to be most valuable in the feature engineering part, the number of features can be greatly reduced. Additionally, `transformations` can be limited to only those feature transformations that make sense for your data. Last but not least, you can subsample the data used for training the model to limit the memory requirements. After the model was fit, you can call `transform()` on your whole dataset to generate only those few features that were selected during `fit()`/`fit_transform()`.


### Installation
You can either download the code from here and include the autofeat folder in your `$PYTHONPATH` or install (the library components only) via pip:

    $ pip install autofeat

The library requires Python 3! Other dependencies: `numpy`, `pandas`, `scikit-learn`, `sympy`, `joblib`, `pint` and `numba`.

