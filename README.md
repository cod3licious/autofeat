# `autofeat` library
### A Linear Regression Model with Automatic Feature Engineering and Selection

This library contains the `AutoFeatRegression` model with a similar interface as the `scikit-learn` models:
- `fit()` function to fit the model parameters
- `predict()` function to predict the target variable given the input
- `score()` function to calculate the goodness of the fit (R^2 value)
- `fit_transform()` and `transform()` functions, which extend the given data by the additional features that were engineered and selected by the model

When calling the `fit()` function, internally the `fit_transform()` function will be called, so if you're planing to call `transform()` on the same data anyways, just call `fit_transform()` right away. `transform()` is mostly useful if you've split your data into training and test data and did not call `fit_transform()` on your whole dataset. The `predict()` and `score()` functions can be either be given data in the format of the original dataframe that was used when calling `fit()`/`fit_transform()` or they can be given an already transformed dataframe.

The [notebook](https://github.com/cod3licious/autofeat/blob/master/autofeat_test.ipynb) contains a simple usage example - try it out! :)

For further details on the model and implementation please refer to the [paper](https://arxiv.org/abs/1901.07329)  - and of course if any of this code was helpful for your research, please consider citing it:
```
    @article{horn2019autofeat,
      author    = {Horn, Franziska and Pack, Robert and Rieger, Michael},
      title     = {The autofeat Python Library for Automatic Feature Engineering and Selection},
      year      = {2019},
      journal   = {arXiv preprint arXiv:1901.07329},
    }
```

The code is intended for research purposes.

If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!

## Installation
You either download the code from here and include the autofeat folder in your `$PYTHONPATH` or install (the library components only) via pip:

    $ pip install autofeat

The library requires Python 3! Other dependencies: `numpy`, `pandas`, `scikit-learn`, `sympy`, and `pint`

## Acknowledgments

This project was made possible thanks to support by the [BASF](https://www.basf.com).
