# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Calibration: a power function of the predicted speedup, or its inverse.

    forward:  max(c0 * speedup^c1 + c2, floor)
    inverse:  max(((speedup - c2) / c0)^(1 / c1), floor)

The forward direction applies a curve that was fitted from predicted to measured speedups.
The inverse direction applies a curve that was fitted from measured to predicted speedups.
"""

import numpy as np
import pandas as pd

DIRECTIONS = ('forward', 'inverse')


def calibrate(
    speedup: pd.Series,
    c0: float = 1.0,
    c1: float = 1.0,
    c2: float = 0.0,
    direction: str = 'forward',
    floor: float = 0.01,
) -> pd.Series:
    if direction not in DIRECTIONS:
        raise ValueError(f'direction must be one of {DIRECTIONS}, got: {direction}')
    if floor <= 0:
        raise ValueError(f'floor must be positive, got: {floor}')

    if direction == 'forward':
        calibrated = c0 * np.power(speedup, c1) + c2
    else:
        if c0 == 0 or c1 == 0:
            raise ValueError(f'The inverse direction needs non-zero c0 and c1, got: c0={c0}, c1={c1}')

        # Below the curve's range the inverse is undefined. Such speedups take the floor.
        base = ((speedup - c2) / c0).clip(lower=0.0)
        calibrated = np.power(base, 1.0 / c1)

    return calibrated.clip(lower=floor)
