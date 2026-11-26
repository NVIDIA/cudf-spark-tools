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

from types import SimpleNamespace

import pytest  # pylint: disable=import-error
import pandas as pd

from spark_rapids_tools.tools.qualx import qualx_main
from spark_rapids_tools.tools.qualx.qualx_main import (
    _get_model_path,
    _get_model,
    _compute_summary,
    _add_entries_for_missing_apps,
)
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
            'Duration': [100, 100, 200],
            'Duration_pred': [50, 50, 100],
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

    @pytest.mark.parametrize('runtimes, model, expected_variant', [
        (['PHOTON'], None, 'PHOTON'),
        (['SPARK'], None, 'SPARK'),
        (['PHOTON'], 'databricks-azure', None),
    ])
    def test_predict_single_runtime_selects_model_variant(self, monkeypatch, runtimes, model, expected_variant):
        '''predict() should load the runtime-specific model also when all inputs share one runtime'''
        class _Stop(Exception):
            pass

        calls = {}

        def fake_get_model(platform, model_arg=None, variant=None):
            calls['model'] = (platform, model_arg, variant)
            return object()

        def fake_get_calib_params(platform, model_arg=None, variant=None):
            calls['calib'] = (platform, model_arg, variant)
            raise _Stop()

        qual_tool_output = pd.DataFrame({'appId': ['app-1'], 'appName': ['app one']})
        profile_df = pd.DataFrame({
            'appId': ['app-1'] * len(runtimes),
            'appName': ['app one'] * len(runtimes),
            'runType': ['CPU'] * len(runtimes),
            'sparkRuntime': runtimes,
        })
        handler = SimpleNamespace(is_empty=lambda: False)
        monkeypatch.setattr(qualx_main, '_get_combined_qual_data',
                            lambda _handlers: (None, qual_tool_output, qual_tool_output))
        monkeypatch.setattr(qualx_main, 'create_row_with_default_speedup', lambda row: row)
        monkeypatch.setattr(qualx_main, 'load_profiles', lambda **_kwargs: profile_df)
        monkeypatch.setattr(qualx_main, '_get_model', fake_get_model)
        monkeypatch.setattr(qualx_main, '_get_calib_params', fake_get_calib_params)

        with pytest.raises(_Stop):
            qualx_main.predict('databricks-azure', 'qual_dir', {}, model=model, qual_handlers=[handler])

        expected = ('databricks-azure', model, expected_variant)
        assert calls['model'] == expected
        assert calls['calib'] == expected
