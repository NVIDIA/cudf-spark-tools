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

"""Test the GPU Eligible column of SpeedupCategory"""

import pandas as pd
import pytest  # pylint: disable=import-error

from spark_rapids_pytools.common.prop_manager import YAMLPropertiesContainer
from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_tools.tools.speedup_category import SpeedupCategory
from ..conftest import SparkRapidsToolsUT

SPEEDUP = 'Estimated GPU Speedup'
UNSUPPORTED_PERCENT = 'Unsupported Operators Stage Duration Percent'
SKIP_BY_HEURISTICS = 'Skip by Heuristics'


def _speedup_category() -> SpeedupCategory:
    conf = YAMLPropertiesContainer(prop_arg=str(Utils.resource_path('qualification-conf.yaml')))
    return SpeedupCategory(conf.get_value('local', 'output', 'speedupCategories'))


def _apps(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[SPEEDUP, UNSUPPORTED_PERCENT, SKIP_BY_HEURISTICS, 'Spark Runtime'])


class TestSpeedupCategoryEligibility(SparkRapidsToolsUT):
    """Test class for the GPU Eligible column"""

    @pytest.mark.parametrize(('row', 'eligible'), [
        ((1.1, 5.0, False, 'SPARK'), True),
        ((4.0, 25.0, False, 'SPARK'), True),
        ((4.0, 40.0, False, 'SPARK'), False),
        ((4.0, 5.0, True, 'SPARK'), False),
        ((0.5, 5.0, False, 'PHOTON'), True),
    ])
    def test_eligibility_ignores_only_the_speedup_bound(self, row, eligible):
        """Every eligibility condition but the speedup bound applies, and so do heuristics"""
        apps = _speedup_category().build_eligibility_column(_apps([row]))

        assert apps['GPU Eligible'].tolist() == [eligible]

    def test_eligible_apps_above_the_bound_are_recommended(self):
        """Above Tools' speedup bound, an app is GPU Eligible exactly when its category is recommended"""
        rows = [(speedup, unsupported, skipped, 'SPARK')
                for speedup in (1.3, 2.5, 4.0)
                for unsupported in (0.0, 25.0, 30.0)
                for skipped in (False, True)]
        speedup_category = _speedup_category()

        apps = speedup_category.build_category_column(_apps(rows))
        apps = speedup_category.build_eligibility_column(apps)

        recommended = apps['Estimated GPU Speedup Category'] != 'Not Recommended'
        assert apps['GPU Eligible'].tolist() == recommended.tolist()

    def test_empty_apps(self):
        """An empty frame gains an empty column"""
        apps = _speedup_category().build_eligibility_column(_apps([]))

        assert 'GPU Eligible' in apps.columns
        assert apps.empty
