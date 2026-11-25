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

"""Test speedup_category module"""

import copy

import pandas as pd
import pytest  # pylint: disable=import-error

from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_tools.tools.speedup_category import (
    SpeedupCategory,
    validate_speedup_threshold,
    with_speedup_threshold,
)
from spark_rapids_tools.utils.propmanager import AbstractPropContainer
from ..conftest import SparkRapidsToolsUT

SPEEDUP_COLUMN = 'Estimated GPU Speedup'
UNSUPPORTED_COLUMN = 'Unsupported Operators Stage Duration Percent'
SPEEDUP_REASON = 'GPU SpeedUp < Qualifying Threshold (More is qualified)'


def speedup_category_props() -> dict:
    """The speedup categories of qualification-conf.yaml, with its default bounds. The bounds of the eligibility
    conditions are strings, as they are read from the environment."""
    def strategy(not_applicable_bound: float) -> dict:
        return {
            'categories': [
                {'title': 'Not Applicable', 'lowerBound': -1000000.0, 'upperBound': not_applicable_bound},
                {'title': 'Small', 'lowerBound': not_applicable_bound, 'upperBound': 2.0},
                {'title': 'Medium', 'lowerBound': 2.0, 'upperBound': 3.0},
                {'title': 'Large', 'lowerBound': 3.0, 'upperBound': 1000000.0},
            ],
            'eligibilityConditions': [
                {'columnName': SPEEDUP_COLUMN, 'lowerBound': str(not_applicable_bound), 'upperBound': '1000000.0',
                 'skippingReason': SPEEDUP_REASON},
                {'columnName': UNSUPPORTED_COLUMN, 'lowerBound': '0.0', 'upperBound': '25.0',
                 'skippingReason': f'{UNSUPPORTED_COLUMN} > Qualifying Threshold (Less is qualified)'},
            ],
        }

    return {
        'speedupColumnName': SPEEDUP_COLUMN,
        'categoryColumnName': 'Estimated GPU Speedup Category',
        'heuristicsColumnName': 'Skip by Heuristics',
        'sparkRuntimeColumnName': 'Spark Runtime',
        'defaultCategory': 'Not Recommended',
        'strategies': {'spark': strategy(1.3), 'photon': strategy(1.0)},
    }


def assert_speedup_threshold_applied(props: dict, original: dict, threshold: float) -> None:
    """Check that props are the original ones, but for each runtime's lower bound of the condition on the speedup,
    and for the bounds of its categories: they start at the threshold, and a configured bound under the threshold
    is raised to it."""
    assert {key: value for key, value in props.items() if key != 'strategies'} == {
        key: value for key, value in original.items() if key != 'strategies'}
    assert list(props['strategies']) == list(original['strategies'])
    medium, large = max(2.0, threshold), max(3.0, threshold)
    for runtime, strategy in original['strategies'].items():
        new_strategy = props['strategies'][runtime]
        assert new_strategy['categories'] == [
            {'title': 'Not Applicable', 'lowerBound': -1000000.0, 'upperBound': threshold},
            {'title': 'Small', 'lowerBound': threshold, 'upperBound': medium},
            {'title': 'Medium', 'lowerBound': medium, 'upperBound': large},
            {'title': 'Large', 'lowerBound': large, 'upperBound': 1000000.0},
        ]
        assert len(new_strategy['eligibilityConditions']) == len(strategy['eligibilityConditions'])
        speedup_conditions = 0
        for condition, new_condition in zip(strategy['eligibilityConditions'], new_strategy['eligibilityConditions']):
            if condition['columnName'] == SPEEDUP_COLUMN:
                speedup_conditions += 1
                assert new_condition == {**condition, 'lowerBound': threshold}
            else:
                assert new_condition == condition
        assert speedup_conditions == 1


class TestSpeedupCategory(SparkRapidsToolsUT):
    """Test class for speedup_category module"""

    def test_validate_speedup_threshold(self):
        """Test validate_speedup_threshold accepts positive finite numbers and rejects invalid values."""
        assert validate_speedup_threshold(1) == 1.0
        assert validate_speedup_threshold(0.8) == 0.8
        assert validate_speedup_threshold('1.5') == 1.5

        for invalid in [0, 0.0, -1.5, float('inf'), float('-inf'), float('nan'), True, False, 'high', []]:
            with pytest.raises(ValueError, match='speedup_threshold must be positive and finite'):
                validate_speedup_threshold(invalid)

    def test_with_speedup_threshold(self):
        """Test that the lower bound of the condition on the speedup and the category bounds change, for every
        runtime"""
        props = speedup_category_props()
        original = copy.deepcopy(props)

        result = with_speedup_threshold(props, 1.5)

        assert_speedup_threshold_applied(result, original, 1.5)
        # the given properties are not changed, then or by a change to the result
        assert props == original
        result['strategies']['spark']['categories'][0]['title'] = 'Changed'
        result['strategies']['photon']['eligibilityConditions'][1]['upperBound'] = '50.0'
        assert props == original

        # thresholds under every configured bound, and above some or all of the other bounds
        for threshold in [0.5, 1.0, 2.5, 4.0]:
            assert_speedup_threshold_applied(with_speedup_threshold(props, threshold), original, threshold)

    def test_with_speedup_threshold_of_packaged_config(self):
        """Test the threshold on the speedup categories of the packaged qualification-conf.yaml"""
        config = AbstractPropContainer.load_from_file(str(Utils.resource_path('qualification-conf.yaml')))
        props = config.get_value('local', 'output', 'speedupCategories')
        original = copy.deepcopy(props)
        assert set(original['strategies']) == {'spark', 'photon'}

        result = with_speedup_threshold(props, 1.5)

        assert_speedup_threshold_applied(result, original, 1.5)
        assert props == original

    def test_with_speedup_threshold_without_conditions_or_categories(self):
        """Test properties without strategies, and strategies without a condition on the speedup or categories"""
        assert not with_speedup_threshold({}, 1.5)
        props = speedup_category_props()
        props['strategies']['spark'].pop('eligibilityConditions')
        del props['strategies']['photon']['eligibilityConditions'][0]
        props['strategies']['photon'].pop('categories')

        result = with_speedup_threshold(props, 1.5)

        assert result['strategies']['photon'] == props['strategies']['photon']
        assert set(result['strategies']['spark']) == {'categories'}
        assert [category['upperBound'] for category in result['strategies']['spark']['categories']] == [
            1.5, 2.0, 3.0, 1000000.0]

    @pytest.mark.parametrize('threshold,speedups,categories', [
        # the categories start at the threshold, and keep their other bounds
        (1.5, [1.49, 1.5, 1.99, 2.0, 2.99, 3.0], ['Not Recommended', 'Small', 'Small', 'Medium', 'Medium', 'Large']),
        # a threshold above the bound of Medium: no app is Small, and a recommended app is Medium at the least
        (2.5, [2.49, 2.5, 2.99, 3.0, 3.5, 9.0], ['Not Recommended', 'Medium', 'Medium', 'Large', 'Large', 'Large']),
        # a threshold above every bound: a recommended app is Large
        (4.0, [2.5, 3.5, 3.99, 4.0, 4.5, 9.0], ['Not Recommended'] * 3 + ['Large'] * 3),
    ])
    def test_categories_start_at_the_threshold(self, threshold, speedups, categories):
        """Test that the categories start at the threshold and keep their other bounds, for every runtime"""
        props = with_speedup_threshold(speedup_category_props(), threshold)
        apps = pd.DataFrame({
            'App ID': ['app1', 'app2', 'app3', 'app4', 'app5', 'app6'],
            'Spark Runtime': ['SPARK', 'PHOTON', 'SPARK', 'PHOTON', 'SPARK', 'PHOTON'],
            SPEEDUP_COLUMN: speedups,
            UNSUPPORTED_COLUMN: [0.0] * 6,
            'Skip by Heuristics': [False] * 6,
            'Not Recommended Reason': [''] * 6,
        })

        result = SpeedupCategory(props).build_category_column(apps)

        assert result['Estimated GPU Speedup Category'].tolist() == categories

    @pytest.mark.parametrize('threshold,categories,reasons', [
        # the configured bounds: 1.3 for Spark and 1.0 for Photon
        (None, ['Small', 'Medium', 'Small', 'Not Recommended'], ['', '', '', SPEEDUP_REASON]),
        (1.5, ['Not Recommended', 'Medium', 'Not Recommended', 'Not Recommended'],
         [SPEEDUP_REASON, '', SPEEDUP_REASON, SPEEDUP_REASON]),
        (1.3, ['Small', 'Medium', 'Small', 'Not Recommended'], ['', '', '', SPEEDUP_REASON]),
        # a threshold under the configured bounds: the categories start at it, so no app is 'Not Applicable'
        (1.0, ['Small', 'Medium', 'Small', 'Small'], ['', '', '', '']),
        (0.5, ['Small', 'Medium', 'Small', 'Small'], ['', '', '', '']),
    ])
    def test_build_category_column(self, threshold, categories, reasons):
        """Test that an app is recommended against the threshold, in place of the configured bound, and that its
        category starts at the threshold"""
        props = speedup_category_props()
        if threshold is not None:
            props = with_speedup_threshold(props, threshold)
        apps = pd.DataFrame({
            'App ID': ['app1', 'app2', 'app3', 'app4'],
            'Spark Runtime': ['SPARK', 'SPARK', 'PHOTON', 'SPARK'],
            SPEEDUP_COLUMN: [1.4, 2.5, 1.4, 1.2],
            UNSUPPORTED_COLUMN: [0.0, 10.0, 0.0, 0.0],
            'Skip by Heuristics': [False, False, False, False],
            'Not Recommended Reason': ['', '', '', ''],
        })

        result = SpeedupCategory(props).build_category_column(apps)

        assert result['App ID'].tolist() == ['app1', 'app2', 'app3', 'app4']
        assert result['Estimated GPU Speedup Category'].tolist() == categories
        assert result['Not Recommended Reason'].tolist() == reasons

    def test_qualification_applies_speedup_threshold_arg(self):
        """Test that Qualification applies wrapper_options['speedupThreshold'] regardless of estimation model."""
        from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
        from spark_rapids_pytools.rapids.qualification import Qualification  # pylint: disable=import-outside-toplevel

        props = speedup_category_props()
        original = copy.deepcopy(props)

        class _FakeCtxt:
            def __init__(self):
                self._values = {'estimationModelArgs': {'xgboostEnabled': False}}

            def set_ctxt(self, key, value):
                self._values[key] = value

            def get_ctxt(self, key):
                return self._values.get(key)

        fake_qual = SimpleNamespace(
            wrapper_options={'speedupThreshold': 1.5},
            ctxt=_FakeCtxt(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
        Qualification._process_speedup_threshold_arg(fake_qual)  # pylint: disable=protected-access
        applied = Qualification._Qualification__apply_speedup_threshold(fake_qual, props)  # pylint: disable=protected-access
        assert_speedup_threshold_applied(applied, original, 1.5)

        fake_qual.wrapper_options = {}
        Qualification._process_speedup_threshold_arg(fake_qual)  # pylint: disable=protected-access
        unchanged = Qualification._Qualification__apply_speedup_threshold(fake_qual, props)  # pylint: disable=protected-access
        assert unchanged == original

        for invalid in [0, -1.0, float('inf'), float('nan'), True, False, 'high']:
            fake_qual.wrapper_options = {'speedupThreshold': invalid}
            with pytest.raises(ValueError, match='speedup_threshold must be positive and finite'):
                Qualification._process_speedup_threshold_arg(fake_qual)  # pylint: disable=protected-access
