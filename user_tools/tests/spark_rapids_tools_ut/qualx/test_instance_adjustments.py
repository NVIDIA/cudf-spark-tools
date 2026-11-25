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

"""Test instance adjustments and per-instance speedup estimates"""

import pandas as pd
import pytest  # pylint: disable=import-error
from pydantic import ValidationError

from spark_rapids_tools.tools.qualx.instance_adjustments import InstanceAdjustments
from spark_rapids_tools.tools.qualx.qualx_main import (
    _add_entries_for_missing_apps,
    _compute_summary,
    estimate_instances,
)
from ..conftest import SparkRapidsToolsUT

G6 = 'g6.4xlarge'
G4DN = 'g4dn.4xlarge'
G5 = 'g5.4xlarge'


def _adjustments(level: str = 'sql', factor: float = 0.5) -> InstanceAdjustments:
    """The model was trained on g6.4xlarge; g4dn.4xlarge gets `factor` of its estimate."""
    return InstanceAdjustments(adjustments={
        G6: {'type': 'linear', 'level': 'sql', 'factor': 1.0},
        G4DN: {'type': 'linear', 'level': level, 'factor': factor},
    })


def _predictions():
    """Per-SQL and per-app predictions for one predicted app (app1) and one app QualX did not predict (app2).

    app1 runs 1000 ms: q1 is 600 ms, all supported, predicted 3x; q2 is 300 ms, half supported, predicted 2x.
    Its predicted duration is 1000 - 900 + 600 / 3 + (150 + 150 / 2) = 525 ms, a 1.905x speedup.
    """
    per_sql = pd.DataFrame({
        'appName': ['app1', 'app1'],
        'appId': ['app1', 'app1'],
        'appDuration': [1000, 1000],
        'sqlID': [1, 2],
        'scaleFactor': [1, 1],
        'Duration': [600, 300],
        'description': ['q1', 'q2'],
        'y_pred': [3.0, 2.0],
        'Duration_pred': [200.0, 225.0],
        'Duration_supported': [600.0, 150.0],
    })
    defaults = pd.DataFrame({
        'appName': ['app1', 'app2'],
        'appId': ['app1', 'app2'],
        'appDuration': [1000, 2000],
        'Duration': [0, 0],
        'Duration_pred': [0, 0],
        'Duration_supported': [0, 0],
        'fraction_supported': [0.0, 0.0],
        'appDuration_pred': [1000, 2000],
        'speedup': [1.0, 1.0],
    })
    per_app = _add_entries_for_missing_apps(defaults, _compute_summary(per_sql))
    return per_sql, per_app


def _row(estimates: pd.DataFrame, app_id: str, instance_type: str) -> pd.Series:
    rows = estimates[(estimates['appId'] == app_id) & (estimates['instanceType'] == instance_type)]
    assert len(rows) == 1
    return rows.iloc[0]


class TestInstanceAdjustments(SparkRapidsToolsUT):
    """Test class for instance adjustments"""

    def test_load_from_file(self, tmp_path):
        """An adjustments file loads, and describes how each estimate is derived"""
        path = tmp_path / 'instances.yaml'
        path.write_text(
            'adjustments:\n'
            '  g6.4xlarge: {type: linear, level: sql, factor: 1.0}\n'
            '  g4dn.4xlarge: {type: linear, level: sql, factor: 0.7}\n',
            encoding='utf-8')

        adjustments = InstanceAdjustments.load_from_file(str(path))

        assert adjustments.describe(G6) == 'linear(1.0, sql)'
        assert adjustments.describe(G4DN) == 'linear(0.7, sql)'
        assert adjustments.adjustments[G6].is_identity
        assert not adjustments.adjustments[G4DN].is_identity
        adjustments.check_targets([G6, G4DN])
        with pytest.raises(ValueError, match='No adjustment for g5.4xlarge'):
            adjustments.check_targets([G4DN, G5])

    @pytest.mark.parametrize('adjustment', [
        {'type': 'linear', 'level': 'app', 'factor': 1.2},
        {'type': 'linear', 'level': 'sql', 'factor': 0},
        {'type': 'linear', 'level': 'query', 'factor': 0.5},
        {'type': 'power', 'level': 'sql', 'factor': 0.5},
        {'type': 'linear', 'level': 'sql', 'factor': 0.5, 'offset': 1},
    ])
    def test_invalid_adjustment(self, adjustment):
        """App-level scale-ups, non-positive factors, and unknown fields are rejected"""
        with pytest.raises(ValidationError):
            InstanceAdjustments(adjustments={G4DN: adjustment})

    @pytest.mark.parametrize(('level', 'factor'), [('sql', 1.6), ('app', 1.0)])
    def test_valid_adjustment(self, level, factor):
        """A factor above 1 is allowed per SQL query, and a factor of 1.0 at either level"""
        InstanceAdjustments(adjustments={G6: {'type': 'linear', 'level': level, 'factor': factor}})

    def test_file_wide_fields_are_rejected(self):
        """Each instance type's entry says how its estimate is derived; there is no file-wide reference"""
        with pytest.raises(ValidationError):
            InstanceAdjustments(reference_instance_type=G6, adjustments={})


class TestEstimateInstances(SparkRapidsToolsUT):
    """Test class for per-instance speedup estimates"""

    def test_identity_adjustment_keeps_the_model_estimate(self):
        """The instance type the model was trained on gets the per-app prediction unchanged"""
        per_sql, per_app = _predictions()

        estimates = estimate_instances(per_sql, per_app, [G6], _adjustments())

        row = _row(estimates, 'app1', G6)
        assert row['speedup'] == pytest.approx(1000 / 525)
        assert row['appDuration_pred'] == 525
        assert row['status'] == 'ok'
        assert row['source'] == 'linear(1.0, sql)'

    def test_sql_level_adjustment_scales_only_supported_time(self):
        """q1 runs 600 / 1.5 = 400 ms and q2 runs 150 + 150 / 1.0 = 300 ms, so app1 runs 800 ms"""
        per_sql, per_app = _predictions()

        row = _row(estimate_instances(per_sql, per_app, [G6, G4DN], _adjustments('sql', 0.5)), 'app1', G4DN)

        assert row['speedup'] == pytest.approx(1000 / 800)
        assert row['appDuration_pred'] == 800
        assert row['status'] == 'ok'
        assert row['source'] == 'linear(0.5, sql)'

    def test_app_level_adjustment_scales_the_app_speedup(self):
        """app1's 1.905x becomes 0.952x, so it runs 1050 ms"""
        per_sql, per_app = _predictions()

        row = _row(estimate_instances(per_sql, per_app, [G4DN], _adjustments('app', 0.5)), 'app1', G4DN)

        assert row['speedup'] == pytest.approx(0.5 * 1000 / 525)
        assert row['appDuration_pred'] == 1050
        assert row['source'] == 'linear(0.5, app)'

    @pytest.mark.parametrize('level', ['sql', 'app'])
    def test_app_without_a_prediction_fails_on_every_instance(self, level):
        """An app QualX did not predict has no estimate on any instance type"""
        per_sql, per_app = _predictions()

        estimates = estimate_instances(per_sql, per_app, [G6, G4DN], _adjustments(level))

        for instance_type in [G6, G4DN]:
            row = _row(estimates, 'app2', instance_type)
            assert row['status'] == 'failed'
            assert pd.isna(row['speedup'])
            assert pd.isna(row['appDuration_pred'])

    def test_sql_level_adjustment_without_per_sql_predictions_fails(self):
        """Without per-SQL predictions, a SQL-level adjustment other than 1.0 cannot be applied"""
        _, per_app = _predictions()

        estimates = estimate_instances(None, per_app, [G6, G4DN], _adjustments('sql'))

        assert _row(estimates, 'app1', G6)['status'] == 'ok'
        assert _row(estimates, 'app1', G4DN)['status'] == 'failed'

    def test_no_targets_or_predictions_give_no_rows(self):
        """Nothing is estimated without target instance types or per-app predictions"""
        per_sql, per_app = _predictions()

        assert estimate_instances(per_sql, per_app, [], None).empty
        assert estimate_instances(per_sql, None, [G6], _adjustments()).empty
        assert estimate_instances(per_sql, pd.DataFrame(), [G6], _adjustments()).empty

    def test_target_without_an_adjustment_is_an_error(self):
        """Every target instance type must have an adjustment"""
        per_sql, per_app = _predictions()

        with pytest.raises(ValueError, match='No adjustment for g5.4xlarge'):
            estimate_instances(per_sql, per_app, [G6, G5], _adjustments())
