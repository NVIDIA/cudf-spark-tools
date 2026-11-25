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

"""Test qualx_preprocess module"""
import os
import json
import tempfile
import types
from unittest.mock import patch

import pandas as pd
import pytest  # pylint: disable=import-error

from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.preprocess import (
    _supports_level,
    expected_raw_features,
    impute,
    get_alignment,
    get_featurizers,
    get_modifiers,
    infer_app_meta,
    load_profiles,
    load_qtool_execs,
    load_datasets
)
from ..conftest import SparkRapidsToolsUT

# a config with the required settings, and the plugins that a test is about
CONFIG = """
datasets: datasets
featurizers: {featurizers}
modifiers: {modifiers}
split_functions: split_train_val.py
model_type: xgboost
xgboost: {{}}
"""


def write_config(path, featurizers=('default.py',), modifiers=()) -> str:
    """Write a config with the given plugins to the path, and return the path as get_config takes it."""
    path.write_text(CONFIG.format(featurizers=list(featurizers), modifiers=list(modifiers)), encoding='utf-8')
    return str(path)


class TestPreprocess(SparkRapidsToolsUT):
    """Test class for qualx_preprocess module"""
    def test_impute(self):
        # Test impute function
        input_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        expected_features = expected_raw_features()
        imputed_df = impute(input_df, expected_features)
        df_columns = set(imputed_df.columns)

        # should not have extra columns
        assert 'col1' not in df_columns
        assert 'col2' not in df_columns

        # should have all expected raw features
        assert df_columns == expected_features

        # fraction_supported should be 1.0
        assert imputed_df['fraction_supported'].iloc[0] == 1.0

        # all other columns should be 0.0
        assert imputed_df[list(df_columns - {'fraction_supported'})].iloc[0].sum() == 0.0

    def test_get_alignment(self):
        # Test get_alignment function
        with patch('spark_rapids_tools.tools.qualx.preprocess.get_config') as mock_config:
            # Create temporary alignment CSV file
            test_alignment = pd.DataFrame(
                [
                    ['app1', 'sql1', 'app3', 'sql3'],
                    ['app2', 'sql2', 'app4', 'sql4']
                ],
                columns=['appId_cpu', 'sqlID_cpu', 'appId_gpu', 'sqlID_gpu'],
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                alignment_file = os.path.join(temp_dir, 'alignment.csv')
                test_alignment.to_csv(alignment_file, index=False)

                mock_config.return_value.alignment_dir = temp_dir

                # Call function
                align_df = get_alignment()

                # Verify results
                assert isinstance(align_df, pd.DataFrame)
                assert len(align_df) == 2
                assert list(align_df.columns) == ['appId_cpu', 'sqlID_cpu', 'appId_gpu', 'sqlID_gpu']
                assert list(align_df['appId_cpu']) == ['app1', 'app2']
                assert list(align_df['sqlID_cpu']) == ['sql1', 'sql2']
                assert list(align_df['appId_gpu']) == ['app3', 'app4']
                assert list(align_df['sqlID_gpu']) == ['sql3', 'sql4']

    def test_get_alignment_app_id_only(self):
        # Test get_alignment function
        with patch('spark_rapids_tools.tools.qualx.preprocess.get_config') as mock_config:
            # Create temporary alignment CSV file
            test_alignment = pd.DataFrame(
                [
                    ['app1', 'app3'],
                    ['app2', 'app4']
                ],
                columns=['appId_cpu', 'appId_gpu'],
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                alignment_file = os.path.join(temp_dir, 'alignment.csv')
                test_alignment.to_csv(alignment_file, index=False)

                mock_config.return_value.alignment_dir = temp_dir

                # Call function
                align_df = get_alignment()

                # Verify results
                assert isinstance(align_df, pd.DataFrame)
                assert len(align_df) == 2
                assert list(align_df.columns) == ['appId_cpu', 'appId_gpu']
                assert list(align_df['appId_cpu']) == ['app1', 'app2']
                assert list(align_df['appId_gpu']) == ['app3', 'app4']

    def test_get_featurizers(self):
        # Test get_featurizers function
        get_config(reload=True)
        result = get_featurizers()

        # Verify results
        assert isinstance(result, list)
        assert len(result) >= 2
        assert all(callable(f.extract_raw_features) for f in result)

        # Test caching
        with patch('spark_rapids_tools.tools.qualx.util.load_plugin') as mock_load_plugin:
            result2 = get_featurizers()
            mock_load_plugin.assert_not_called()
            assert result == result2

    def test_get_modifiers(self):
        # Test get_modifiers function
        with patch('spark_rapids_tools.tools.qualx.preprocess.get_config') as mock_config:
            mock_config.return_value.modifiers = ['align_sql_id.py']
            result = get_modifiers()

            # Verify results
            assert isinstance(result, list)
            assert len(result) >= 1
            assert all(callable(f.modify) for f in result)

            # Test caching
            with patch('spark_rapids_tools.tools.qualx.util.load_plugin') as mock_load_plugin:
                result2 = get_modifiers()
                mock_load_plugin.assert_not_called()
                assert result == result2

    def test_get_featurizers_follows_config(self, tmp_path):
        """Test that the featurizers are cached for the loaded config, and loaded again after the config is"""
        one_path = write_config(tmp_path / 'one.yaml', featurizers=['default.py'])
        two_path = write_config(tmp_path / 'two.yaml', featurizers=['default.py', 'hash_plan.py'])
        try:
            get_config(one_path)
            one = get_featurizers()
            assert [f.__name__ for f in one] == ['default']
            assert 'hash' not in expected_raw_features()

            # the same config: the cached list, without loading the plugins again
            with patch('spark_rapids_tools.tools.qualx.preprocess.load_plugin') as mock_load_plugin:
                assert get_featurizers() is one
                get_config(one_path)
                assert get_featurizers() is one
                mock_load_plugin.assert_not_called()

            # a config from another path: its featurizers, without a reload
            get_config(two_path)
            two = get_featurizers()
            assert two is not one
            assert [f.__name__ for f in two] == ['default', 'hash_plan']
            assert get_featurizers() is two
            assert 'hash' in expected_raw_features()

            # and back to the first config, which is loaded again
            get_config(one_path)
            again = get_featurizers()
            assert again is not one
            assert [f.__name__ for f in again] == ['default']
            assert get_featurizers(reload=True) is not again
        finally:
            get_config(reload=True)  # reload config
        assert [f.__name__ for f in get_featurizers()] == ['default', 'hash_plan']

    def test_get_modifiers_follows_config(self, tmp_path):
        """Test that the modifiers are cached for the loaded config, and loaded again after the config is"""
        with_path = write_config(tmp_path / 'with.yaml', modifiers=['align_sql_id.py'])
        without_path = write_config(tmp_path / 'without.yaml')
        try:
            get_config(with_path)
            modifiers = get_modifiers()
            assert [m.__name__ for m in modifiers] == ['align_sql_id']
            assert get_modifiers() is modifiers

            get_config(without_path)
            assert get_modifiers() == []

            get_config(with_path)
            again = get_modifiers()
            assert again is not modifiers
            assert [m.__name__ for m in again] == ['align_sql_id']
        finally:
            get_config(reload=True)  # reload config
        assert get_modifiers() == []

    def test_supports_level(self):
        """Test that the default featurizer has app-level features, and the plan hash doesn't"""
        get_config(reload=True)
        assert {f.__name__: _supports_level(f) for f in get_featurizers()} == {'default': True, 'hash_plan': False}

    def test_load_profiles_level(self, tmp_path, monkeypatch):
        """Test that app-level profiles come from the featurizers that take a level, and merge on the app"""
        calls = []

        def main_features(toc, node_level_supp, qualtool_filter, qualtool_output=None, remove_failed_sql=True,
                          level='sql'):
            # pylint: disable=unused-argument
            calls.append(('main', level, toc['appId'].unique().tolist()))
            if level == 'app':
                return pd.DataFrame({'appId': ['app-1', 'app-2'], 'sqlID': -1, 'appName': 'ds', 'total': [10, 5]})
            return pd.DataFrame({'appId': ['app-1', 'app-1', 'app-2'], 'sqlID': [0, 1, 0], 'appName': 'ds',
                                 'total': [4, 6, 5]})

        def sql_only_features(toc, node_level_supp, qualtool_filter, qualtool_output=None, remove_failed_sql=True):
            # pylint: disable=unused-argument
            calls.append(('sql_only',))
            return pd.DataFrame({'appId': ['app-1', 'app-1', 'app-2'], 'sqlID': [0, 1, 0], 'hash': ['a', 'b', 'c']})

        def extra_features(toc, node_level_supp, qualtool_filter, qualtool_output=None, remove_failed_sql=True,
                           level='sql'):
            # pylint: disable=unused-argument
            calls.append(('extra', level))
            if level == 'app':
                # keyed by the app alone, and with a SQL id that is not the main featurizer's
                return pd.DataFrame({'appId': ['app-2', 'app-1'], 'extra': [7, 8]})
            return pd.DataFrame({'appId': ['app-1', 'app-1', 'app-2'], 'sqlID': [1, 0, 0], 'extra': [1, 2, 3]})

        main = types.SimpleNamespace(
            __name__='main', extract_raw_features=main_features, supported_levels=('sql', 'app'),
            expected_raw_features={'appId', 'sqlID', 'appName', 'total', 'runType', 'scaleFactor', 'platform_onprem'})
        sql_only = types.SimpleNamespace(
            __name__='sql_only', extract_raw_features=sql_only_features, expected_raw_features={'hash'})
        extra = types.SimpleNamespace(
            __name__='extra', extract_raw_features=extra_features, supported_levels=('sql', 'app'),
            expected_raw_features={'extra'})
        featurizers = [main, sql_only, extra]
        get_config(reload=True)
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.preprocess.get_featurizers', lambda: featurizers)

        # the profiler output of two apps
        for app_id in ['app-1', 'app-2']:
            (tmp_path / app_id).mkdir()
            (tmp_path / app_id / 'application_information.csv').write_text('appId\n', encoding='utf-8')
        datasets = {'ds': {'profiles': [str(tmp_path)], 'app_meta': {'default': {'runType': 'CPU', 'scaleFactor': 1}}}}

        # by default, one row per SQL from every featurizer, which are not passed a level
        sql_df = load_profiles(datasets).sort_values(['appId', 'sqlID'])
        assert [call[0] for call in calls] == ['main', 'sql_only', 'extra']
        assert calls[0] == ('main', 'sql', ['app-1', 'app-2'])
        assert sql_df[['appId', 'sqlID']].values.tolist() == [['app-1', 0], ['app-1', 1], ['app-2', 0]]
        assert sql_df['total'].tolist() == [4, 6, 5]
        assert sql_df['hash'].tolist() == ['a', 'b', 'c']
        assert sql_df['extra'].tolist() == [2, 1, 3]
        pd.testing.assert_frame_equal(load_profiles(datasets, level='sql').sort_values(['appId', 'sqlID']), sql_df)

        # one row per app, from the featurizers that take a level
        calls.clear()
        app_df = load_profiles(datasets, level='app').sort_values('appId')
        assert calls == [('main', 'app', ['app-1', 'app-2']), ('extra', 'app')]
        assert app_df['appId'].tolist() == ['app-1', 'app-2']
        assert app_df['sqlID'].tolist() == [-1, -1]
        assert app_df['total'].tolist() == [10, 5]
        assert app_df['extra'].tolist() == [8, 7]
        # the feature of the skipped featurizer is imputed
        assert app_df['hash'].tolist() == [0, 0]
        assert set(app_df.columns) == set(sql_df.columns)
        assert (app_df[['runType', 'scaleFactor', 'platform_onprem']] == ['CPU', 1, 1]).all(axis=None)

        # with no featurizer that takes a level, there are no app-level profiles
        featurizers[:] = [sql_only]
        with pytest.raises(ValueError, match='None of the featurizers can extract app-level features'):
            load_profiles(datasets, level='app')

        for level in ['stage', 'job', None]:
            with pytest.raises(ValueError, match='level must be one of'):
                load_profiles(datasets, level=level)

    def test_load_profiles_app_level_is_unfiltered(self, tmp_path, monkeypatch):
        """Test that the qualification tool filter applies to the SQL-level profiles only: the app-level profiles
        are those of the filter 'none', whatever the filter is"""
        calls = []

        def features(toc, node_level_supp, qualtool_filter, qualtool_output=None, remove_failed_sql=True,
                     level='sql'):
            # pylint: disable=unused-argument
            calls.append((level, qualtool_filter))
            if level == 'app':
                rows = pd.DataFrame({'appId': ['app-1', 'app-2'], 'sqlID': -1, 'appName': 'ds', 'total': [10, 6]})
            else:
                rows = pd.DataFrame({'appId': ['app-1', 'app-1', 'app-2'], 'sqlID': [0, 1, 0], 'appName': 'ds',
                                     'total': [4, 6, 6]})
            if qualtool_filter in ('stage', 'sqlId'):
                # a filter keeps the supported half of each row, and tells the fraction that it kept
                rows = rows.assign(total=rows['total'] // 2, fraction_supported=0.5)
            return rows

        featurizer = types.SimpleNamespace(
            __name__='main', extract_raw_features=features, supported_levels=('sql', 'app'),
            expected_raw_features={'appId', 'sqlID', 'appName', 'total', 'fraction_supported', 'runType',
                                   'scaleFactor', 'platform_onprem'})
        get_config(reload=True)
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.preprocess.get_featurizers', lambda: [featurizer])

        # the profiler output of two apps
        for app_id in ['app-1', 'app-2']:
            (tmp_path / app_id).mkdir()
            (tmp_path / app_id / 'application_information.csv').write_text('appId\n', encoding='utf-8')
        datasets = {'ds': {'profiles': [str(tmp_path)], 'app_meta': {'default': {'runType': 'CPU', 'scaleFactor': 1}}}}

        # the SQL-level profiles are filtered as before
        for qual_tool_filter in ['stage', 'sqlId']:
            sql_df = load_profiles(datasets, qual_tool_filter=qual_tool_filter)
            assert sql_df['total'].tolist() == [2, 3, 3]
            assert sql_df['fraction_supported'].tolist() == [0.5, 0.5, 0.5]
        for qual_tool_filter in ['none', None]:
            sql_df = load_profiles(datasets, qual_tool_filter=qual_tool_filter)
            assert sql_df['total'].tolist() == [4, 6, 6]
            assert sql_df['fraction_supported'].tolist() == [1.0, 1.0, 1.0]
        assert calls == [('sql', 'stage'), ('sql', 'sqlId'), ('sql', 'none'), ('sql', None)]

        # the app-level profiles cover the whole app, and the featurizer is not passed the filter
        calls.clear()
        app_df = load_profiles(datasets, qual_tool_filter='none', level='app')
        assert app_df['total'].tolist() == [10, 6]
        assert app_df['fraction_supported'].tolist() == [1.0, 1.0]
        for qual_tool_filter in ['stage', 'sqlId', None]:
            other_df = load_profiles(datasets, qual_tool_filter=qual_tool_filter, level='app')
            pd.testing.assert_frame_equal(other_df, app_df)
        assert calls == [('app', 'none')] * 4

    def test_infer_app_meta(self):
        # Test infer_app_meta function
        with patch('spark_rapids_tools.tools.qualx.preprocess.find_eventlogs') as mock_find_eventlogs:
            # Setup mock eventlogs
            mock_find_eventlogs.return_value = [
                '/path/to/job1/eventlogs/CPU/app1',
                '/path/to/job1/eventlogs/GPU/app2',
                '/path/to/job2/eventlogs/CPU/app3',
                '/path/to/job2/eventlogs/GPU/app4',
            ]

            # Call function
            result = infer_app_meta(['/path/to/job1', '/path/to/job2'])

            # Verify results
            assert isinstance(result, dict)
            assert len(result) == 4
            assert result['app1'] == {
                'jobName': 'job1',
                'runType': 'CPU',
                'scaleFactor': 1
            }
            assert result['app2'] == {
                'jobName': 'job1',
                'runType': 'GPU',
                'scaleFactor': 1
            }
            assert result['app3'] == {
                'jobName': 'job2',
                'runType': 'CPU',
                'scaleFactor': 1
            }
            assert result['app4'] == {
                'jobName': 'job2',
                'runType': 'GPU',
                'scaleFactor': 1
            }

    def test_load_qtool_execs(self):
        # Test load_qtool_execs function
        # Create test data
        test_data = pd.DataFrame([
            ['app1', 'sql1', 'node1', True, '', 'Exec1'],
            ['app1', 'sql2', 'node2', False, 'IgnoreNoPerf', 'Exec2'],
            ['app2', 'sql1', 'node1', False, '', 'WholeStageCodegen3'],
            ['app2', 'sql1', 'node2', False, '', 'Exec4']
        ], columns=['App ID', 'SQL ID', 'SQL Node Id', 'Exec Is Supported', 'Action', 'Exec Name'])
        # Call function
        result = load_qtool_execs(test_data)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['App ID', 'SQL ID', 'SQL Node Id', 'Exec Is Supported']
        assert len(result) == 4
        assert result['Exec Is Supported'].tolist() == [True, True, True, False]

    def test_load_datasets(self):
        """Test load_datasets function"""
        get_config(reload=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test dataset structure
            dataset_dir = os.path.join(temp_dir, 'datasets', 'onprem')
            os.makedirs(dataset_dir)

            # Create test JSON file
            test_json = {
                'dataset1': {
                    'eventlogs': ['/path/to/eventlog1'],
                    'platform': 'onprem',
                    'app_meta': {
                        'app1': {
                            'runType': 'CPU',
                            'scaleFactor': 1
                        }
                    }
                }
            }

            json_path = os.path.join(dataset_dir, 'dummy.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(test_json, f)

            # Create dummy preprocessed.parquet cache file
            test_profile = pd.DataFrame([
                ['dataset1', 'app1', 'sql1', 100, 0.8],
                ['dataset1', 'app1', 'sql2', 150, 0.6],
                ['dataset2', 'app2', 'sql1', 200, 0.9],
                ['dataset2', 'app2', 'sql2', 180, 0.7]
            ], columns=['appName', 'appId', 'sqlID', 'Duration', 'fraction_supported'])
            profile_dir = os.path.join(get_config().cache_dir, 'onprem')
            os.makedirs(profile_dir, exist_ok=True)
            test_profile.to_parquet(os.path.join(profile_dir, 'preprocessed.parquet'), index=False)

            # Call function
            datasets, profile_df = load_datasets(dataset_dir)

            # Verify results
            assert isinstance(datasets, dict)
            assert 'dataset1' in datasets
            assert datasets['dataset1']['platform'] == 'onprem'

            assert isinstance(profile_df, pd.DataFrame)
            assert not profile_df.empty
            assert len(profile_df) == 2  # only 2 rows correspond to dataset1
