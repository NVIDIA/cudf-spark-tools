# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Test qualx_main module"""

import pytest  # pylint: disable=import-error
import pandas as pd

from spark_rapids_tools.tools.qualx.qualx_main import (
    _get_model_path,
    _get_model,
    _compute_summary,
    _add_entries_for_missing_apps,
)
from spark_rapids_tools.tools.qualx.rollup import CostFunction
from ..conftest import SparkRapidsToolsUT


class TestMain(SparkRapidsToolsUT):
    """Test class for qualx_main module"""

    def test_get_model_path_with_platform(self):
        """Test _get_model_path with platform only"""
        platform = 'onprem'
        model_path = _get_model_path(platform, None)
        assert model_path.name == 'onprem.json'
        assert model_path.exists()

    def test_get_model_path_with_model(self):
        '''Test _get_model_path with explicit model name'''
        platform = 'onprem'
        model = 'onprem'
        model_path = _get_model_path(platform, model)
        assert model_path.name == 'onprem.json'
        assert model_path.exists()

    def test_get_model_path_with_variant(self):
        '''Test _get_model_path with platform variant'''
        platform = 'onprem'
        variant = 'PHOTON'
        model_path = _get_model_path(platform, None, variant)
        # Should fall back to base platform model if variant model doesn't exist
        assert model_path.name == 'onprem.json'
        assert model_path.exists()

    def test_get_model_path_invalid_platform(self):
        '''Test _get_model_path with invalid platform'''
        platform = 'invalid_platform'
        with pytest.raises(ValueError, match=r'Platform \[invalid_platform\] does not have a pre-trained model'):
            _get_model_path(platform, None)

    def test_get_model_with_platform(self):
        '''Test _get_model with platform only'''
        platform = 'onprem'
        model = _get_model(platform)
        assert model is not None
        assert hasattr(model, 'predict')

    def test_compute_summary(self):
        """Test _compute_summary with sample data"""
        # Create sample input data
        data = {
            'appName': ['app1', 'app1', 'app2'],
            'appId': ['id1', 'id1', 'id2'],
            'appDuration': [100, 100, 200],
            'appWallTime': [100, 100, 200],
            'Duration': [100, 100, 200],
            'Duration_pred': [50, 50, 100],
            'fraction_supported': [1.0, 1.0, 1.0],
            'y_pred': [2.0, 2.0, 2.0],
            'Duration_supported': [100, 100, 200],
            'sqlDuration': [50, 50, 100],
            'sqlDuration_pred': [25, 25, 50],
            'sqlDuration_supported': [50, 50, 100],
            'description': ['desc1', 'desc1', 'desc2'],
            'scaleFactor': [1.0, 1.0, 1.0]
        }
        results = pd.DataFrame(data)

        summary = _compute_summary(results)

        # Verify summary calculations
        assert len(summary) == 2  # Should have 2 unique apps
        assert all(col in summary.columns for col in [
            'appName', 'appId', 'appDuration', 'fraction_supported',
            'appDuration_pred', 'speedup'
        ])
        # app1's SQLs overlap, so its duration is raised to their sum
        assert summary['appDuration'].tolist() == [200, 200]
        assert summary['appDuration_pred'].tolist() == [100, 100]
        assert summary['speedup'].tolist() == [2.0, 2.0]

    def test_compute_summary_of_task_time(self, monkeypatch):
        """Test _compute_summary with a cost of task time, and with prices"""
        cost_fn = CostFunction.load('duration_sum.py', price_ratio=0.5)
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.qualx_main.get_cost_function', lambda: cost_fn)
        # one app of two SQLs that hold 3,000 of its 4,000 ms of task time in 60 of its 100 s, and are a quarter and
        # a half supported
        results = pd.DataFrame({
            'appName': 'app1',
            'appId': 'id1',
            'appDuration': 100_000,
            'appWallTime': 100_000,
            'appTaskTime': 4000,
            'scaleFactor': 1.0,
            'description': 'desc1',
            'Duration': [20_000, 40_000],
            'duration_sum': [1000, 2000],
            'fraction_supported': [0.25, 0.5],
            'y_pred': [2.0, 2.0],
        })
        assert cost_fn.label == 'duration_sum'
        results['duration_sum_supported'] = results['duration_sum'] * results['fraction_supported']
        results['duration_sum_pred'] = results['duration_sum'] - results['duration_sum_supported'] / 2

        summary = _compute_summary(results).iloc[0]

        # the supported share of the app's task time: the SQLs count for 3/4 of the app, and not for 60% of it
        assert summary['fraction_supported'] == pytest.approx((250 + 1000) / 4000)
        # the app's durations are its costs, as whole numbers
        assert summary['cost'] == 4000
        assert summary['appDuration'] == 4000
        assert summary['gpuCost'] == pytest.approx(4000 - (250 + 1000) / 2)
        assert summary['appDuration_pred'] == int(summary['gpuCost'])
        # the speedup is at the prices, and the costs aren't
        assert summary['speedup'] == pytest.approx(0.5 * 4000 / summary['gpuCost'])
        assert summary['duration_sum'] == 3000
        assert summary['duration_sum_supported'] == 1250

    def test_add_entries_for_missing_apps(self):
        """Test _add_entries_for_missing_apps"""
        # Create sample data
        all_default_preds = pd.DataFrame({
            'appId': ['id1', 'id2', 'id3'],
            'appName': ['app1', 'app2', 'app3'],
            'appDuration': [100, 200, 300]
        })

        summary_df = pd.DataFrame({
            'appId': ['id1', 'id2'],
            'appName': ['app1', 'app2'],
            'appDuration': [100, 200],
            'speedup': [2.0, 1.5]
        })

        result = _add_entries_for_missing_apps(all_default_preds, summary_df)

        # Verify results
        assert len(result) == 3  # Should include all apps
        assert 'wasPredicted' in result.columns
        assert sum(result['wasPredicted']) == 2  # Two apps were predicted
        assert not result.loc[result['appId'] == 'id3', 'wasPredicted'].iloc[0]  # id3 was not predicted
