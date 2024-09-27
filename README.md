# Autofeat

**Autofeat** is a Python library that provides `sklearn`-compatible linear prediction models with automated feature engineering and selection capabilities.

## Overview

Autofeat simplifies the process of improving linear model performance by automating feature generation and selection. It first generates a wide range of non-linear features, then selects a small, robust subset of meaningful features that enhance the predictive power of linear models. This multi-step approach allows you to harness the interpretability of linear models without sacrificing accuracy.

### Key Features:
- **Automated Feature Generation and Selection**: Automates the process of generating and selecting features for linear models for improved performance.
- **Improved Performance and Interpretability**: The generated features improve prediction accuracy while retaining the intuitive interpretability of linear models.
- **Seamless Integration**: Fully compatible with `scikit-learn` pipelines, making it easy to integrate into your existing machine learning workflows.

### Use Cases:
- Ideal for **supervised learning tasks** where model transparency is crucial for decision-making.
- Suitable for **feature selection** in large datasets, automating the discovery of important variables.
- Useful in scenarios where **non-linear features** need to be discovered and leveraged without complicating the model.

**Note:** The code is intended for research purposes. Results may vary depending on the dataset and use case.

## Installation

Autofeat is available on PyPI, making it easy to install via `pip`:

```
pip install autofeat
```
### Other Dependencies 
- numpy 
- pandas
- scikit-learn
- sympy
- joblib
- pint
- numba

## Documentation and Resources
| Description | Link |
|-------------|------|
| Example Notebooks | [examples](/notebooks/) |
| Documentation | [documentation](https://franziskahorn.de/autofeat) |


If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!

## Acknowledgments
This project was made possible thanks to support by [BASF](https://www.basf.com).

