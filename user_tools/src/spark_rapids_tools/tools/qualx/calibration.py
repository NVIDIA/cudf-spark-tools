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
Calibration of predicted app speedups.

A calibration maps the raw app speedup that the model predicts to a calibrated speedup. It can be used to
correct for model bias by adjusting the predictions coming out of the model. The config selects one with
`calibration`:

    calibration:
      path: power.py     # a built-in under calibrations/, or the path of a plugin file
      args: {c0: 0.51, c1: 1.0, c2: 2.52, direction: inverse, floor: 0.5}

A plugin is a Python file with one function:

    def calibrate(speedup: pd.Series, **args) -> pd.Series:
        '''Map predicted speedups to calibrated speedups.'''
"""

import inspect
import os
import types
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.util import load_plugin

_calibration = None  # global cache of the calibration
_calibration_config = None  # the config that the calibration was loaded for


def _resolve_plugin_path(path: str, config_dir: Optional[str] = None) -> str:
    """Resolve a calibration plugin path."""
    expanded = os.path.expandvars(path)
    if os.path.isabs(expanded) and os.path.exists(expanded):
        return expanded

    qualx_dir = os.path.dirname(__file__)
    search_dirs = [qualx_dir] + ([config_dir] if config_dir else []) + [os.getcwd()]
    for subdir in ('calibrations', 'plugins', ''):
        for base in search_dirs:
            candidate = os.path.join(base, subdir, expanded)
            if os.path.exists(candidate):
                return candidate
    raise ValueError(f'{expanded} not found')


class Calibration:
    """A calibration plugin, bound to its args."""

    def __init__(self, plugin: types.ModuleType, args: Optional[dict] = None):
        if not callable(getattr(plugin, 'calibrate', None)):
            raise ValueError(f'Calibration {plugin.__name__} must define calibrate(speedup, **args)')

        self.name = plugin.__name__
        self.args = dict(args or {})
        self._calibrate = plugin.calibrate

        # Check the args against the plugin when it is loaded.
        try:
            inspect.signature(self._calibrate).bind(pd.Series(dtype=float), **self.args)
        except TypeError as e:
            raise ValueError(f'Calibration {self.name} does not accept the args {self.args}: {e}') from e

    @classmethod
    def load(
        cls,
        path: str,
        args: Optional[dict] = None,
        *,
        config_dir: Optional[str] = None,
    ) -> 'Calibration':
        """Load a built-in calibration by file name, or a plugin by path."""
        plugin = load_plugin(_resolve_plugin_path(path, config_dir=config_dir))
        return cls(plugin, args)

    def apply(self, speedup: pd.Series) -> pd.Series:
        """Calibrate speedups, keeping their index."""
        speedup = pd.Series(speedup)
        calibrated = self._calibrate(speedup, **self.args)
        if not isinstance(calibrated, pd.Series) or not calibrated.index.equals(speedup.index):
            raise ValueError(f'Calibration {self.name} must return a Series with the index of its input')

        values = calibrated.to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f'Calibration {self.name} must return finite, positive speedups')

        return calibrated


def calibrate_speedups(
    speedups: pd.Series,
    calibration: Optional[Mapping[str, Any]],
    *,
    config_dir: Optional[str] = None,
) -> pd.Series:
    """
    Apply a calibration spec (`{'path': ..., 'args': ...}`) to speedups directly.
    Returns a copy of `speedups` as a Series when `calibration` is None or empty.
    This is a stateless alternative to loading the calibration from the global config.
    """
    if not calibration:
        return pd.Series(speedups).copy()
    spec = dict(calibration)
    if 'path' not in spec or set(spec) - {'path', 'args'}:
        raise ValueError(f'Calibration spec must have path and optional args, got: {sorted(spec)}')
    return Calibration.load(spec['path'], spec.get('args'), config_dir=config_dir).apply(speedups)


def get_calibration() -> Optional[Calibration]:
    """
    Lazily load the calibration that the config selects, using global cache. Returns None if it does not use one.
    The cache is rebuilt when the config is reloaded.
    """
    global _calibration, _calibration_config  # pylint: disable=global-statement
    config = get_config()
    if _calibration_config is not config:
        if config.calibration:
            config_dir = os.path.dirname(config.file_path) if config.file_path else None
            _calibration = Calibration.load(**dict(config.calibration), config_dir=config_dir)
        else:
            _calibration = None
        _calibration_config = config
    return _calibration
