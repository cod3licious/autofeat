# `autofeat` library
### A Linear Regression Model with Automatic Feature Engineering and Selection

This library contains the `AutoFeatRegression` model with a similar interface as the `scikit-learn` models:
- `fit()` function to fit the model parameters
- `predict()` function to predict the target variable given the input
- `score()` function to calculate the goodness of the fit (R^2 value)
- `fit_transform()` and `transform()` functions, which extend the given data by the additional features that were engineered and selected by the model

When calling the `fit()` function, internally the `fit_transform()` function will be called, so if you're planing to call `transform()` on the same data anyways, just call `fit_transform()` right away. `transform()` is mostly useful if you've split your data into training and test data and did not call `fit_transform()` on your whole dataset. The `predict()` and `score()` functions can either be given data in the format of the original dataframe that was used when calling `fit()`/`fit_transform()` or they can be given an already transformed dataframe.

In addition, only the feature selection part is also available in the `FeatureSelector` model.

Both the `AutoFeatRegression` and `FeatureSelector` models need to be **fit on data without NaNs**, as they internally call the sklearn `LassoLarsCV` model, which can not handle NaNs. When calling `transform()`, NaNs (but not `np.inf`) are okay.

The [autofeat examples notebook](https://github.com/cod3licious/autofeat/blob/master/autofeat_examples.ipynb) contains a simple usage example - try it out! :) Additional examples can be found in the [autofeat benchmark notebook](https://github.com/cod3licious/autofeat/blob/master/autofeat_benchmark.ipynb) (which also contains the code to reproduce the results from the paper mentioned below) as well as the testing scripts.

Please keep in mind that since the `AutoFeatRegression` model can generate very complex features, it will likely **overfit on noise** in the dataset, though the coefficients for features related to noise should be fairly small. It is generally suggested to carefully inspect the features found by `autofeat` and use those that make sense to you to train your own models.

Depending on the number of `feateng_steps` (default 2) and the number of input features, `autofeat` can generate a very huge feature matrix (before selecting the most appropriate features from this large feature pool). By specifying in `feateng_cols` those columns that you expect to be most valuable in the feature engineering part, the number of features can be greatly reduced. Additionally, `transformations` can be limited to only those feature transformations that make sense for your data. Last but not least, you can subsample the data used for training the model to limit the memory requirements. After the model was fit, you can call `transform()` on your whole dataset to generate only those few features that were selected during `fit()`/`fit_transform()`.

#### What if I have a (binary) classification problem?

(UNTESTED!) While the `autofeat` library is intended for regression problems (with a single target variable), you could also use the model to generate additional features for a binary classification problem (i.e. where your target variable is a 1D vector with only a 0 or 1 for each data point). After calling `fit_transform()` on the `AutoFeatRegression` model to generate the additional features, these can be used as input to your favorite classification model to get the actual class label predictions. It may happen that during the model-fit you get an error due to numerical instabilities. In this case you can try using as targets the class probabilities (i.e. what you get when calling `predict_proba()` on a `sklearn` classifier) instead of the binary labels.

Since the `AutoFeatRegression` model only works for a single target variable, it can't handle multi-class problems, but you could transform a multi-class problem into several 1-vs-rest binary classification problems and then generate features for each of these individually.


## Installation
You either download the code from here and include the autofeat folder in your `$PYTHONPATH` or install (the library components only) via pip:

    $ pip install autofeat

The library requires Python 3! Other dependencies: `numpy`, `pandas`, `scikit-learn`, `sympy`, `joblib`, and `pint`


## Paper
For further details on the model and implementation please refer to the [paper](https://arxiv.org/abs/1901.07329)  - and of course if any of this code was helpful for your research, please consider citing it:
```
    @article{horn2019autofeat,
      author    = {Horn, Franziska and Pack, Robert and Rieger, Michael},
      title     = {The autofeat Python Library for Automated Feature Engineering and Selection},
      year      = {2019},
      journal   = {arXiv preprint arXiv:1901.07329},
    }
```

The code is intended for research purposes.

If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!

## Acknowledgments

This project was made possible thanks to support by [BASF](https://www.basf.com).
