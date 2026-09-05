# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Pure Python/NumPy Fallback for Qlib Expanding Operators.
Provides 100% API compatibility when compiled Cython C-extensions (.pyd) are unavailable.
"""

import math
import numpy as np


class Expanding:
    """1-D array expanding base class"""

    def __init__(self):
        self.na_count = 0
        self.barv = []

    def update(self, val: float) -> float:
        raise NotImplementedError


class Mean(Expanding):
    """1-D array expanding mean"""

    def __init__(self):
        super().__init__()
        self.vsum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        if math.isnan(val):
            self.na_count += 1
        else:
            self.vsum += val
        valid = len(self.barv) - self.na_count
        return self.vsum / valid if valid > 0 else float("nan")


class Slope(Expanding):
    """1-D array expanding slope"""

    def __init__(self):
        super().__init__()
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.y_sum = 0.0
        self.xy_sum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        size = float(len(self.barv))
        if math.isnan(val):
            self.na_count += 1
        else:
            self.x_sum += size
            self.x2_sum += size * size
            self.y_sum += val
            self.xy_sum += size * val

        N = len(self.barv) - self.na_count
        denom = N * self.x2_sum - self.x_sum * self.x_sum
        if N <= 1 or denom == 0:
            return float("nan")
        return (N * self.xy_sum - self.x_sum * self.y_sum) / denom


class Resi(Expanding):
    """1-D array expanding residuals"""

    def __init__(self):
        super().__init__()
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.y_sum = 0.0
        self.xy_sum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        size = float(len(self.barv))
        if math.isnan(val):
            self.na_count += 1
        else:
            self.x_sum += size
            self.x2_sum += size * size
            self.y_sum += val
            self.xy_sum += size * val

        N = len(self.barv) - self.na_count
        denom = N * self.x2_sum - self.x_sum * self.x_sum
        if N <= 1 or denom == 0:
            return float("nan")
        slope = (N * self.xy_sum - self.x_sum * self.y_sum) / denom
        x_mean = self.x_sum / N
        y_mean = self.y_sum / N
        interp = y_mean - slope * x_mean
        return val - (slope * size + interp)


class Rsquare(Expanding):
    """1-D array expanding rsquare"""

    def __init__(self):
        super().__init__()
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.y_sum = 0.0
        self.y2_sum = 0.0
        self.xy_sum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        size = float(len(self.barv))
        if math.isnan(val):
            self.na_count += 1
        else:
            self.x_sum += size
            self.x2_sum += size * size
            self.y_sum += val
            self.y2_sum += val * val
            self.xy_sum += size * val

        N = len(self.barv) - self.na_count
        denom_x = N * self.x2_sum - self.x_sum * self.x_sum
        denom_y = N * self.y2_sum - self.y_sum * self.y_sum
        if N <= 1 or denom_x <= 0 or denom_y <= 0:
            return float("nan")
        rvalue = (N * self.xy_sum - self.x_sum * self.y_sum) / math.sqrt(denom_x * denom_y)
        return rvalue * rvalue


def _expanding(r: Expanding, a: np.ndarray) -> np.ndarray:
    N = len(a)
    ret = np.empty(N, dtype=np.float64)
    for i in range(N):
        ret[i] = r.update(float(a[i]))
    return ret


def expanding_mean(a: np.ndarray) -> np.ndarray:
    return _expanding(Mean(), a)


def expanding_slope(a: np.ndarray) -> np.ndarray:
    return _expanding(Slope(), a)


def expanding_rsquare(a: np.ndarray) -> np.ndarray:
    return _expanding(Rsquare(), a)


def expanding_resi(a: np.ndarray) -> np.ndarray:
    return _expanding(Resi(), a)

