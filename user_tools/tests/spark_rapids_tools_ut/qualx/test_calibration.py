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

"""Test calibration module"""

import types

import numpy as np
import pandas as pd
import pytest  # pylint: disable=import-error

from spark_rapids_tools.tools.qualx import config as qualx_config_mod
from spark_rapids_tools.tools.qualx.calibration import Calibration, calibrate_speedups, get_calibration
from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.util import get_abs_path, load_plugin
from ..conftest import SparkRapidsToolsUT

# A calibration plugin whose result is chosen by its mode, for the checks that apply makes on a result.
PLUGIN = """
import numpy as np


def calibrate(speedup, mode='scaled', factor=2.0):
    if mode == 'scaled':
        return speedup * factor
    if mode == 'reindexed':
        return (speedup * factor).reset_index(drop=True)
    if mode == 'shorter':
        return speedup.iloc[1:]
    if mode == 'array':
        return speedup.to_numpy() * factor
    values = {'nan': np.nan, 'inf': np.inf, 'zero': 0.0, 'negative': -1.0}
    return speedup * values[mode]
"""

# a config with the required settings, to which a test adds a calibration
BASE_CONFIG = """
datasets: datasets
featurizers: ['default.py']
split_functions: split_train_val.py
model_type: xgboost
xgboost: {}
"""


@pytest.fixture(name='power')
def _power():
    """The built-in power calibration, with the default config loaded."""
    get_config(reload=True)
    return load_plugin(get_abs_path('power.py', 'calibrations'))


@pytest.fixture(name='plugin_path')
def _plugin_path(tmp_path):
    """A calibration plugin outside the package, with the default config loaded."""
    get_config(reload=True)
    path = tmp_path / 'scaled.py'
    path.write_text(PLUGIN, encoding='utf-8')
    return str(path)


class TestPowerCalibration(SparkRapidsToolsUT):
    """Test class for the built-in power calibration"""

    def test_forward(self, power):
        speedup = pd.Series([0.5, 1.0, 2.0, 4.0], index=[3, 5, 7, 11])
        calibrated = power.calibrate(speedup, c0=1.2, c1=1.5, c2=0.1)
        assert calibrated.index.equals(speedup.index)
        np.testing.assert_allclose(calibrated, 1.2 * speedup ** 1.5 + 0.1)
        # forward is the default direction
        forward = power.calibrate(speedup, c0=1.2, c1=1.5, c2=0.1, direction='forward')
        pd.testing.assert_series_equal(calibrated, forward)
        # the default args keep the speedups, as floats
        pd.testing.assert_series_equal(power.calibrate(pd.Series([1, 2, 3])), pd.Series([1.0, 2.0, 3.0]))

    def test_inverse(self, power):
        speedup = pd.Series([0.71, 1.0, 2.52, 6.0], index=[3, 5, 7, 11])
        calibrated = power.calibrate(speedup, c0=0.51, c1=1.3, c2=0.2, direction='inverse')
        assert calibrated.index.equals(speedup.index)
        np.testing.assert_allclose(calibrated, ((speedup - 0.2) / 0.51) ** (1 / 1.3))

    def test_inverse_of_forward(self, power):
        """Test that the two directions undo each other, where neither result is floored"""
        rng = np.random.default_rng(0)
        args = {'c0': 0.51, 'c1': 1.3, 'c2': 0.2, 'floor': 0.01}
        speedup = pd.Series(rng.uniform(0.05, 20.0, 500))
        forward = power.calibrate(speedup, **args, direction='forward')
        np.testing.assert_allclose(power.calibrate(forward, **args, direction='inverse'), speedup, rtol=1e-9)

        # and from the range of the curve, which starts above c2
        in_range = pd.Series(rng.uniform(0.25, 20.0, 500))
        inverse = power.calibrate(in_range, **args, direction='inverse')
        np.testing.assert_allclose(power.calibrate(inverse, **args, direction='forward'), in_range, rtol=1e-9)

    def test_keeps_the_order_of_the_speedups(self, power):
        """Test that a curve with positive c0 and c1 never ranks a lower predicted speedup above a higher one"""
        speedup = pd.Series(np.linspace(0.05, 20.0, 400))
        for direction in ['forward', 'inverse']:
            calibrated = power.calibrate(speedup, c0=0.51, c1=1.3, c2=0.2, direction=direction, floor=0.5)
            assert calibrated.is_monotonic_increasing

    def test_floor(self, power):
        # forward: results under the floor take the floor
        calibrated = power.calibrate(pd.Series([0.1, 0.6, 1.0]), c2=-0.5, floor=0.25)
        assert calibrated.tolist() == [0.25, 0.25, 0.5]
        # inverse: speedups under the range of the curve have no inverse, and take the floor
        calibrated = power.calibrate(pd.Series([0.1, 0.2, 0.3, 1.2]), c0=0.5, c2=0.2, direction='inverse', floor=0.5)
        assert calibrated.tolist() == pytest.approx([0.5, 0.5, 0.5, 2.0])
        # the default floor
        assert power.calibrate(pd.Series([0.0, 0.001])).tolist() == [0.01, 0.01]
        assert power.calibrate(pd.Series([-3.0]), direction='inverse').tolist() == [0.01]

        for floor in [0, -0.5]:
            with pytest.raises(ValueError, match='floor must be positive'):
                power.calibrate(pd.Series([1.0]), floor=floor)

    def test_invalid_args(self, power):
        speedup = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match='direction must be one of'):
            power.calibrate(speedup, direction='backward')
        # the inverse divides by c0, and takes the root of c1
        for args in [{'c0': 0}, {'c1': 0}, {'c0': 0.0, 'c1': 0.0}]:
            with pytest.raises(ValueError, match='needs non-zero c0 and c1'):
                power.calibrate(speedup, **args, direction='inverse')
            assert power.calibrate(speedup, **args, c2=1.5, direction='forward').notna().all()


class TestCalibration(SparkRapidsToolsUT):
    """Test class for calibration module"""

    def test_load(self):
        get_config(reload=True)
        args = {'c0': 0.5, 'c2': 0.2, 'direction': 'inverse', 'floor': 0.5}
        calibration = Calibration.load('power.py', args)
        assert calibration.name == 'power'
        assert calibration.args == args
        # the calibration keeps its own args
        args['floor'] = 0.75
        assert calibration.args['floor'] == 0.5

        speedup = pd.Series([0.3, 1.2, 2.2], index=['app1', 'app2', 'app3'])
        calibrated = calibration.apply(speedup)
        assert calibrated.index.equals(speedup.index)
        assert calibrated.tolist() == pytest.approx([0.5, 2.0, 4.0])

        # without args, the defaults of the plugin
        calibration = Calibration.load('power.py')
        assert not calibration.args
        assert calibration.apply(speedup).tolist() == speedup.tolist()

    def test_load_invalid(self, tmp_path):
        get_config(reload=True)
        # the args are checked against the plugin when it is loaded
        for args in [{'c3': 0.0}, {'c0': 0.5, 'exponent': 2.0}, {'speedup': 1.0}]:
            with pytest.raises(ValueError, match='does not accept the args'):
                Calibration.load('power.py', args)

        with pytest.raises(ValueError, match='not found'):
            Calibration.load('no_such_calibration.py')

        # a plugin without the function
        path = tmp_path / 'no_function.py'
        path.write_text('calibrate = 1.5\n', encoding='utf-8')
        with pytest.raises(ValueError, match='must define calibrate'):
            Calibration.load(str(path))
        with pytest.raises(ValueError, match='must define calibrate'):
            Calibration(types.ModuleType('empty'))

    def test_load_plugin_by_path(self, plugin_path, tmp_path, monkeypatch):
        calibration = Calibration.load(plugin_path, {'factor': 3.0})
        assert calibration.name == 'scaled'
        speedup = pd.Series([0.5, 1.0, 4.0], index=[10, 20, 30])
        pd.testing.assert_series_equal(calibration.apply(speedup), speedup * 3.0)

        # the path can name an environment variable
        monkeypatch.setenv('QUALX_TEST_CALIBRATIONS', str(tmp_path))
        calibration = Calibration.load('${QUALX_TEST_CALIBRATIONS}/scaled.py')
        pd.testing.assert_series_equal(calibration.apply(speedup), speedup * 2.0)

        # the args of this plugin are not those of the built-in
        with pytest.raises(ValueError, match='does not accept the args'):
            Calibration.load(plugin_path, {'c0': 0.5})

    def test_apply(self, plugin_path):
        # speedups that are not a Series are calibrated as one
        calibration = Calibration.load(plugin_path)
        calibrated = calibration.apply(np.array([1.0, 2.5], dtype=np.float32))
        pd.testing.assert_series_equal(calibrated, pd.Series([2.0, 5.0], dtype=np.float32))
        assert calibration.apply(pd.Series(dtype=float)).empty

    @pytest.mark.parametrize('mode', ['nan', 'inf', 'zero', 'negative'])
    def test_apply_rejects_invalid_speedups(self, plugin_path, mode):
        calibration = Calibration.load(plugin_path, {'mode': mode})
        with pytest.raises(ValueError, match='must return finite, positive speedups'):
            calibration.apply(pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30]))

    @pytest.mark.parametrize('mode', ['reindexed', 'shorter', 'array'])
    def test_apply_rejects_another_index(self, plugin_path, mode):
        calibration = Calibration.load(plugin_path, {'mode': mode})
        with pytest.raises(ValueError, match='must return a Series with the index of its input'):
            calibration.apply(pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30]))

    def test_apply_rejects_one_invalid_speedup(self):
        """Test that one speedup the plugin cannot calibrate fails the batch: here, a missing prediction"""
        get_config(reload=True)
        calibration = Calibration.load('power.py', {'c0': 0.5, 'direction': 'inverse'})
        with pytest.raises(ValueError, match='must return finite, positive speedups'):
            calibration.apply(pd.Series([1.0, np.nan, 3.0]))

    def test_get_calibration(self, plugin_path, tmp_path):
        """Test that the calibration is the one of the loaded config, and is loaded again with the config"""
        assert get_calibration() is None

        config_path = tmp_path / 'qualx-conf.yaml'
        config_path.write_text(
            BASE_CONFIG + f'calibration:\n  path: {plugin_path}\n  args: {{factor: 3.0}}\n', encoding='utf-8')
        get_config(str(config_path))
        calibration = get_calibration()
        assert calibration.name == 'scaled'
        assert calibration.args == {'factor': 3.0}
        # the calibration is loaded once for a config
        assert get_calibration() is calibration

        # another config, which names the built-in calibration
        other_path = tmp_path / 'other-conf.yaml'
        other_path.write_text(BASE_CONFIG + 'calibration:\n  path: power.py\n', encoding='utf-8')
        get_config(str(other_path))
        assert get_calibration().name == 'power'

        # and a config without one
        get_config(reload=True)
        assert get_calibration() is None

    def test_calibrate_speedups_is_stateless(self, tmp_path, monkeypatch):
        """calibrate_speedups applies a calibration spec without touching the process-global QualX config."""
        monkeypatch.setattr(qualx_config_mod, '_config', None)
        speedups = pd.Series([0.5, 2.0, 6.0], index=[7, 3, 5])

        # None or empty spec returns a copy unchanged without loading any global config
        unchanged = calibrate_speedups(speedups, None)
        pd.testing.assert_series_equal(unchanged, speedups)
        assert unchanged is not speedups
        assert qualx_config_mod._config is None  # pylint: disable=protected-access

        # Built-in calibration
        spec = {'path': 'power.py', 'args': {'c0': 0.5, 'c2': 0.2, 'direction': 'inverse', 'floor': 0.7}}
        calibrated = calibrate_speedups(speedups, spec)
        assert calibrated.index.tolist() == [7, 3, 5]
        assert calibrated.tolist() == pytest.approx([0.7, 3.6, 11.6])
        assert qualx_config_mod._config is None  # pylint: disable=protected-access

        # Relative plugin resolved via explicit config_dir
        plugin_file = tmp_path / 'scaled_rel.py'
        plugin_file.write_text(PLUGIN, encoding='utf-8')
        rel_calibrated = calibrate_speedups(
            speedups, {'path': 'scaled_rel.py', 'args': {'factor': 3.0}}, config_dir=str(tmp_path)
        )
        pd.testing.assert_series_equal(rel_calibrated, speedups * 3.0)
        assert qualx_config_mod._config is None  # pylint: disable=protected-access

        # Invalid spec / args raise ValueError
        with pytest.raises(ValueError, match='Calibration spec must have path and optional args'):
            calibrate_speedups(speedups, {'args': {'c0': 0.5}})
        with pytest.raises(ValueError, match='does not accept the args'):
            calibrate_speedups(speedups, {'path': 'power.py', 'args': {'exponent': 2.0}})
