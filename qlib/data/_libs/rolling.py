# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Pure Python/NumPy Fallback for Qlib Rolling Operators.
Provides 100% API compatibility when compiled Cython C-extensions (.pyd) are unavailable.
"""

import math
from collections import deque
import numpy as np


class Rolling:
    """1-D array rolling base class"""

    def __init__(self, window: int):
        self.window = int(window)
        self.na_count = self.window
        self.barv = deque([float("nan")] * self.window)

    def update(self, val: float) -> float:
        raise NotImplementedError


class Mean(Rolling):
    """1-D array rolling mean"""

    def __init__(self, window: int):
        super().__init__(window)
        self.vsum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        front = self.barv.popleft()
        if not math.isnan(front):
            self.vsum -= front
        else:
            self.na_count -= 1

        if math.isnan(val):
            self.na_count += 1
        else:
            self.vsum += val

        valid_count = self.window - self.na_count
        return self.vsum / valid_count if valid_count > 0 else float("nan")


class Slope(Rolling):
    """1-D array rolling slope"""

    def __init__(self, window: int):
        super().__init__(window)
        self.i_sum = 0.0
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.y_sum = 0.0
        self.xy_sum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2.0 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum

        front = self.barv.popleft()
        if not math.isnan(front):
            self.i_sum -= 1.0
            self.y_sum -= front
        else:
            self.na_count -= 1

        if math.isnan(val):
            self.na_count += 1
        else:
            self.i_sum += 1.0
            self.x_sum += float(self.window)
            self.x2_sum += float(self.window * self.window)
            self.y_sum += val
            self.xy_sum += float(self.window) * val

        N = self.window - self.na_count
        denom = N * self.x2_sum - self.x_sum * self.x_sum
        if N <= 1 or denom == 0:
            return float("nan")
        return (N * self.xy_sum - self.x_sum * self.y_sum) / denom


class Rsquare(Rolling):
    """1-D array rolling rsquare"""

    def __init__(self, window: int):
        super().__init__(window)
        self.i_sum = 0.0
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.y_sum = 0.0
        self.y2_sum = 0.0
        self.xy_sum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2.0 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum

        front = self.barv.popleft()
        if not math.isnan(front):
            self.i_sum -= 1.0
            self.y_sum -= front
            self.y2_sum -= front * front
        else:
            self.na_count -= 1

        if math.isnan(val):
            self.na_count += 1
        else:
            self.i_sum += 1.0
            self.x_sum += float(self.window)
            self.x2_sum += float(self.window * self.window)
            self.y_sum += val
            self.y2_sum += val * val
            self.xy_sum += float(self.window) * val

        N = self.window - self.na_count
        denom_x = N * self.x2_sum - self.x_sum * self.x_sum
        denom_y = N * self.y2_sum - self.y_sum * self.y_sum
        if N <= 1 or denom_x <= 0 or denom_y <= 0:
            return float("nan")
        rvalue = (N * self.xy_sum - self.x_sum * self.y_sum) / math.sqrt(denom_x * denom_y)
        return rvalue * rvalue


class Resi(Rolling):
    """1-D array rolling residuals"""

    def __init__(self, window: int):
        super().__init__(window)
        self.i_sum = 0.0
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.y_sum = 0.0
        self.xy_sum = 0.0

    def update(self, val: float) -> float:
        self.barv.append(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2.0 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum

        front = self.barv.popleft()
        if not math.isnan(front):
            self.i_sum -= 1.0
            self.y_sum -= front
        else:
            self.na_count -= 1

        if math.isnan(val):
            self.na_count += 1
        else:
            self.i_sum += 1.0
            self.x_sum += float(self.window)
            self.x2_sum += float(self.window * self.window)
            self.y_sum += val
            self.xy_sum += float(self.window) * val

        N = self.window - self.na_count
        denom = N * self.x2_sum - self.x_sum * self.x_sum
        if N <= 1 or denom == 0:
            return float("nan")
        slope = (N * self.xy_sum - self.x_sum * self.y_sum) / denom
        x_mean = self.x_sum / N
        y_mean = self.y_sum / N
        interp = y_mean - slope * x_mean
        return val - (slope * self.window + interp)


def _rolling(r: Rolling, a: np.ndarray) -> np.ndarray:
    N = len(a)
    ret = np.empty(N, dtype=np.float64)
    for i in range(N):
        ret[i] = r.update(float(a[i]))
    return ret


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    return _rolling(Mean(window), a)


def rolling_slope(a: np.ndarray, window: int) -> np.ndarray:
    return _rolling(Slope(window), a)


def rolling_rsquare(a: np.ndarray, window: int) -> np.ndarray:
    return _rolling(Rsquare(window), a)


def rolling_resi(a: np.ndarray, window: int) -> np.ndarray:
    return _rolling(Resi(window), a)

