# -*- coding: utf-8 -*-
# Author: Jeethu Rao <jboloor@acm.org>
# License: MIT

import numba as nb
import numpy as np


@nb.njit(inline='always')
def nb_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit(cache=True)
def nb_nanmean(array, axis):
    return nb_apply_along_axis(np.nanmean, axis, array)


@nb.njit(cache=True)
def nb_nanstd(array, axis):
    return nb_apply_along_axis(np.nanstd, axis, array)


@nb.njit(cache=True)
def nb_standard_scale(array):
    return (array - nb_nanmean(array, 0)) / nb_nanstd(array, 0)
