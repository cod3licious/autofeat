# Author: Franziska Horn <cod3licious@gmail.com>
# License: MIT

name = "autofeat"
__version__ = "2.1.2"
from .autofeatlight import AutoFeatLight  # noqa
from .autofeat import AutoFeatModel, AutoFeatRegressor, AutoFeatClassifier  # noqa
from .featsel import FeatureSelector  # noqa
