# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

"""Test qualx_util module"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest  # pylint: disable=import-error
from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.rollup import CostFunction
from spark_rapids_tools.tools.qualx.util import (
    ensure_directory,
    find_paths,
    find_eventlogs,
    get_dataset_platforms,
    compute_accuracy,
    compute_precision_recall,
    load_plugin,
    random_string,
    print_summary,
    print_speedup_summary,
    create_row_with_default_speedup,
    write_csv_reports,
    log_fallback,
    get_abs_path,
)
from ..conftest import SparkRapidsToolsUT


class TestUtils(SparkRapidsToolsUT):
    """Test class for qualx_util module"""
    def test_ensure_directory(self):
        with patch('os.makedirs') as mock_makedirs:
            ensure_directory('/path/to/dir')
            mock_makedirs.assert_called_once_with(Path('/path/to/dir'), exist_ok=True)

    def test_find_paths(self):
        with patch('os.walk') as mock_walk:
            with patch('os.path.isdir') as mock_isdir:
                mock_walk.return_value = [('/path/to/dir', ['dir1', 'dir2'], ['file1', 'file2'])]
                mock_isdir.return_value = True
                result = find_paths('/path/to/dir')
                assert result == ['/path/to/dir/file1', '/path/to/dir/file2']

    def test_find_eventlogs(self):
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = ['/eventlog/log1', '/eventlog/log2']
            result = find_eventlogs('/eventlog/*.log')
            assert result == ['/eventlog/log1', '/eventlog/log2']

    def test_get_abs_path(self):
        get_config(reload=True)  # reload config

        # Test absolute path case
        with patch('os.path.isabs') as mock_isabs, \
             patch('os.path.exists') as mock_exists:
            mock_isabs.return_value = True
            mock_exists.return_value = True
            result = get_abs_path('/absolute/path')
            assert result == '/absolute/path'

        # Test relative path case
        result = get_abs_path('util.py')
        assert os.path.exists(result) and os.path.isabs(result)

        # Test relative path with subdirectory
        result = get_abs_path('split_all_test.py', 'split_functions')
        assert os.path.exists(result) and os.path.isabs(result)

    def test_get_dataset_platforms(self):
        # supported platforms
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ['databricks-aws', 'onprem']
            result = get_dataset_platforms('/path/to/datasets')
            assert result == (['databricks-aws', 'onprem'], '/path/to/datasets')

        # unsupported platform
        with pytest.raises(ValueError):
            with patch('os.listdir') as mock_listdir:
                mock_listdir.return_value = ['fake_platform']
                result = get_dataset_platforms('/path/to/datasets')

    def test_compute_accuracy(self):
        data = pd.DataFrame({
            'label': [1, 2, 3],
            'pred1': [1, 3, 2],
            'pred2': [2, 1, 3],
            'weight': [1, 2, 3],
        })
        result = compute_accuracy(data, 'label', {'Pred1': 'pred1', 'Pred2': 'pred2'}, 'weight')
        assert result['Pred1']['MAPE'] == pytest.approx(5/18)
        assert result['Pred1']['wMAPE'] == pytest.approx(1/3)
        assert result['Pred1']['dMAPE'] == pytest.approx(1/3)
        assert result['Pred2']['MAPE'] == pytest.approx(0.5)
        assert result['Pred2']['wMAPE'] == pytest.approx(1/3)
        assert result['Pred2']['dMAPE'] == pytest.approx(1/3)

    def test_compute_precision_recall(self):
        data = pd.DataFrame({
            'label': [1, 2, 3],
            'pred1': [1, 3, 2],
            'pred2': [2, 1, 3]
        })
        result = compute_precision_recall(data, 'label', {'Pred1': 'pred1', 'Pred2': 'pred2'}, 1.5)
        expected_result = (
            {'Pred1': 1.0, 'Pred2': 0.5},
            {'Pred1': 1.0, 'Pred2': 0.5}
        )
        assert result == expected_result

    def test_load_plugin(self):
        with pytest.raises(FileNotFoundError):
            load_plugin('/path/to/missing_plugin.py')
        # TODO: load dummy plugin

    def test_random_string(self):
        result = random_string(10)
        assert len(result) == 10

    def test_print_summary(self):
        data = pd.DataFrame({
            'appName': ['app1', 'app2'],
            'appId': ['id1', 'id2'],
            'appDuration': [1, 2],
            'Duration': [1, 2],
            'Duration_pred': [1, 2],
            'Duration_supported': [1, 2],
            'fraction_supported': [1, 2],
            'appDuration_pred': [1, 2],
            'speedup': [1, 2]
        })
        with patch('builtins.print') as mock_print:
            print_summary(data)
            mock_print.assert_called_once()

    def test_print_speedup_summary(self):
        data = pd.DataFrame({
            'appDuration': [1, 2],
            'appDuration_pred': [1, 2]
        })
        with patch('builtins.print') as mock_print:
            print_speedup_summary(data)
            mock_print.assert_called()

        # the overall speedup is at the config's prices
        cost_fn = CostFunction.load('duration.py', price_ratio=0.25)
        with patch('spark_rapids_tools.tools.qualx.util.get_cost_function', return_value=cost_fn), \
                patch('spark_rapids_tools.tools.qualx.util.tabulate') as mock_tabulate, patch('builtins.print'):
            print_speedup_summary(data)
        summary_df = mock_tabulate.call_args.args[0]
        assert summary_df.loc['Overall estimated speedup'].iloc[0] == 0.25

    @pytest.mark.parametrize('label', ['Duration', 'duration_sum'])
    def test_create_row_with_default_speedup(self, label):
        os.environ['QUALX_LABEL'] = label
        get_config(reload=True)  # reload config
        data = pd.Series({
            'App Name': 'app1',
            'App ID': 'id1',
            'App Duration': 1
        })
        result = create_row_with_default_speedup(data)
        expected_result = pd.Series({
            'appName': 'app1',
            'appId': 'id1',
            'appDuration': 1,
            label: 0,
            f'{label}_pred': 0,
            f'{label}_supported': 0,
            'fraction_supported': 0.0,
            'appDuration_pred': 1,
            'speedup': 1.0
        })
        pd.testing.assert_series_equal(result, expected_result)

    def test_write_csv_reports(self):
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            write_csv_reports(
                pd.DataFrame(),
                pd.DataFrame(),
                {'perSql': {'path': '/sql/path'}, 'perApp': {'path': '/app/path'}},
            )
            mock_to_csv.assert_any_call('/app/path')
            mock_to_csv.assert_any_call('/sql/path')

    def test_log_fallback(self):
        logger = MagicMock()
        log_fallback(logger, ['app1', 'app2'], 'reason')
        logger.warning.assert_called_once()
