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

import os
import types

import pytest  # pylint: disable=import-error
import pandas as pd

from spark_rapids_tools.tools.qualx.calibration import Calibration
from spark_rapids_tools.tools.qualx.config import default_config_path, get_config
from spark_rapids_tools.tools.qualx.qualx_main import (
    _get_model_path,
    _get_model,
    _calibrate_summary,
    _compute_summary,
    _compute_app_summary,
    _summarize,
    _add_entries_for_missing_apps,
    _predict_cli,
    _prediction_config,
    featurize,
    predict,
    predict_features,
)
from ..conftest import SparkRapidsToolsUT

QUALX_MAIN = 'spark_rapids_tools.tools.qualx.qualx_main'

# a config with the required settings, to which a test adds the settings that it is about
BASE_CONFIG = """
datasets: datasets
featurizers: ['default.py']
split_functions: split_train_val.py
model_type: xgboost
xgboost: {}
"""


# the settings of a calibration that maps a speedup s to (s - 0.2) / 0.5, and to 0.5 at the least
CALIBRATION = """
calibration:
  path: power.py
  args: {c0: 0.5, c2: 0.2, direction: inverse, floor: 0.5}
"""


def write_config(path, settings: str = '') -> str:
    """Write a config with the given settings to the path, and return the path as predict takes it."""
    path.write_text(BASE_CONFIG + settings, encoding='utf-8')
    return str(path)


def sql_predictions() -> pd.DataFrame:
    """The predictions for the three SQLs of two apps, which roll up to a speedup of 2.0 for each app."""
    return pd.DataFrame({
        'appName': ['app1', 'app1', 'app2'],
        'appId': ['id1', 'id1', 'id2'],
        'appDuration': [100, 100, 200],
        'appWallTime': [100, 100, 200],
        'Duration': [100, 100, 200],
        'Duration_pred': [50, 50, 100],
        'fraction_supported': [1.0, 1.0, 1.0],
        'y_pred': [2.0, 2.0, 2.0],
        'Duration_supported': [100, 100, 200],
        'description': ['desc1', 'desc1', 'desc2'],
        'scaleFactor': [1.0, 1.0, 1.0],
        'speedup_pred': [2.0, 2.0, 2.0],
    })


def app_predictions() -> pd.DataFrame:
    """The predictions for the app-level rows of two apps, of which the second is half supported."""
    return pd.DataFrame({
        'appName': ['app1', 'app2'],
        'appId': ['id1', 'id2'],
        'appDuration': [1000, 3000],
        'sqlID': [-1, -1],
        'scaleFactor': [1, 1],
        'Duration': [800, 2400],
        'fraction_supported': [1.0, 0.5],
        'description': ['', ''],
        'appWallTime': [1000, 3000],
        'y_pred': [2.0, 3.0],
        'Duration_pred': [400.0, 1600.0],
        'Duration_supported': [800.0, 1200.0],
        'speedup_pred': [2.0, 1.5],
    })


@pytest.fixture(name='calibration')
def _calibration():
    """A calibration that maps a speedup s to (s - 0.2) / 0.5, and to 0.5 at the least."""
    get_config(reload=True)
    return Calibration.load('power.py', {'c0': 0.5, 'c2': 0.2, 'direction': 'inverse', 'floor': 0.5})


@pytest.fixture(name='prediction')
def _prediction(monkeypatch, tmp_path):
    """Replace the steps of predict that read a qualification output and run the model.

    The qualification output has three apps, of which the third has no features. Yields the raw features and
    the predictions that the steps return at each level, the calls of the featurizing step as (filter, level),
    the levels that it fails at, and the output files.
    """
    state = types.SimpleNamespace(
        raw_features={
            'sql': pd.DataFrame({'appId': ['id1', 'id1', 'id2'], 'sqlID': [0, 1, 0], 'fraction_supported': 1.0}),
            'app': pd.DataFrame({'appId': ['id1', 'id2'], 'sqlID': [-1, -1], 'fraction_supported': [1.0, 0.5]}),
        },
        predictions={'sql': sql_predictions(), 'app': app_predictions()},
        calls=[],
        failing_levels=set(),
        output_info={
            name: {'path': str(tmp_path / file_name)}
            for name, file_name in {
                'perSql': 'per_sql.csv',
                'perApp': 'per_app.csv',
                'shapValues': 'shap_values.csv',
                'features': 'features.csv',
                'featureImportance': 'feature_importance.csv',
                'rawFeatures': 'raw_features.parquet',
                'rawFeaturesApp': 'raw_features_app.parquet',
            }.items()
        },
    )
    qual_tool_output = pd.DataFrame({
        'App Name': ['app1', 'app2', 'app3'],
        'App ID': ['id1', 'id2', 'id3'],
        'App Duration': [1000, 3000, 500],
    })

    def load_raw_features(platform, qual, qual_data, qual_filter, level='sql'):
        # pylint: disable=unused-argument
        state.calls.append((qual_filter, level))
        if level in state.failing_levels:
            raise ValueError(f'No {level}-level features')
        return state.raw_features[level]

    def predict_raw_features(profile_df, platform, model=None):
        # pylint: disable=unused-argument
        level = 'app' if profile_df is state.raw_features['app'] else 'sql'
        return pd.DataFrame({'feature': [1.0]}), state.predictions[level], 'model'

    monkeypatch.setattr(f'{QUALX_MAIN}._load_qual_data', lambda qual_handlers: (None, qual_tool_output, ['metrics']))
    monkeypatch.setattr(f'{QUALX_MAIN}._load_raw_features', load_raw_features)
    monkeypatch.setattr(f'{QUALX_MAIN}._predict_raw_features', predict_raw_features)
    monkeypatch.setattr(f'{QUALX_MAIN}.compute_shapley_values',
                        lambda xgb_model, features: (pd.DataFrame({'shap_value': [0.0]}), features))
    get_config(reload=True)
    yield state
    get_config(reload=True)  # reload config


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

    def test_compute_app_summary(self, monkeypatch):
        """Test _compute_app_summary with one row per app"""
        get_config(reload=True)
        results = app_predictions()
        results.index = [7, 3]
        unchanged = results.copy()

        summary = _compute_app_summary(results)

        # one row per app, in the order of the input
        assert summary['appId'].tolist() == ['id1', 'id2']
        assert summary.index.tolist() == [0, 1]
        assert summary.columns.tolist() == [
            'appName', 'appId', 'appDuration', 'scaleFactor', 'Duration', 'Duration_pred', 'Duration_supported',
            'description', 'fraction_supported', 'appDuration_pred', 'speedup',
        ]
        # the speedup is the prediction for the row, after its adjustment for unsupported operators
        assert summary['speedup'].tolist() == [2.0, 1.5]
        assert summary['appDuration'].tolist() == [1000, 3000]
        assert summary['appDuration_pred'].tolist() == [500, 2000]
        assert summary['fraction_supported'].tolist() == [1.0, 0.5]
        for col in ['appDuration', 'appDuration_pred', 'Duration', 'Duration_pred', 'Duration_supported']:
            assert pd.api.types.is_integer_dtype(summary[col])
        pd.testing.assert_frame_equal(results, unchanged)

        # the columns of another label
        monkeypatch.setattr(f'{QUALX_MAIN}.get_label', lambda: 'duration_sum')
        label_cols = [col for col in results.columns if col.startswith('Duration')]
        results = results.rename(columns={col: col.replace('Duration', 'duration_sum') for col in label_cols})
        summary = _compute_app_summary(results)
        assert summary[['duration_sum', 'duration_sum_pred', 'duration_sum_supported']].values.tolist() == [
            [800, 400, 800], [2400, 1600, 1200]]
        assert summary['appDuration_pred'].tolist() == [500, 2000]

        # more than one row for an app is not an app-level prediction
        with pytest.raises(ValueError, match='one feature row per app'):
            _compute_app_summary(pd.concat([results, results.iloc[[0]]]))
        with pytest.raises(ValueError, match='one feature row per app'):
            _compute_app_summary(sql_predictions())

    def test_calibrate_summary_without_calibration(self, monkeypatch):
        """Test that the raw speedup is the speedup when the config has no calibration"""
        monkeypatch.setattr(f'{QUALX_MAIN}.get_calibration', lambda: None)
        summary = pd.DataFrame({
            'appId': ['id1', 'id2'],
            'appDuration': [1000, 3000],
            'appDuration_pred': [500, 2000],
            'speedup': [2.0, 1.5],
        })

        result = _calibrate_summary(summary.copy())

        assert result['speedup_raw'].tolist() == [2.0, 1.5]
        pd.testing.assert_frame_equal(result.drop(columns='speedup_raw'), summary)

    def test_calibrate_summary(self, monkeypatch, calibration):
        """Test that a calibration changes the speedup, keeps the raw one and scales the duration"""
        monkeypatch.setattr(f'{QUALX_MAIN}.get_calibration', lambda: calibration)
        summary = pd.DataFrame({
            'appId': ['id1', 'id2', 'id3'],
            'appDuration': [1000, 3000, 900],
            'appDuration_pred': [500, 2000, 3000],
            'speedup': [2.0, 1.5, 0.3],
        })

        result = _calibrate_summary(summary.copy())

        assert result['speedup_raw'].tolist() == [2.0, 1.5, 0.3]
        assert result['speedup'].tolist() == pytest.approx([3.6, 2.6, 0.5])
        # the predicted duration is scaled by speedup_raw / speedup, to a whole number
        assert result['appDuration_pred'].tolist() == [278, 1154, 1800]
        assert pd.api.types.is_integer_dtype(result['appDuration_pred'])
        # so that the durations still give the speedup
        assert result['appDuration'].tolist() == [1000, 3000, 900]
        speedup_of_durations = result['appDuration'] / result['appDuration_pred']
        assert speedup_of_durations.tolist() == pytest.approx(result['speedup'].tolist(), rel=0.01)

        # a summary of no apps
        result = _calibrate_summary(summary.iloc[:0].copy())
        assert result.empty
        assert 'speedup_raw' in result

    def test_calibrate_summary_keeps_an_unusable_speedup(self, monkeypatch, calibration):
        """Test that an app without a finite, positive speedup does not fail the calibration of the other apps"""
        monkeypatch.setattr(f'{QUALX_MAIN}.get_calibration', lambda: calibration)
        summary = pd.DataFrame({
            'appId': ['id1', 'id2', 'id3'],
            'appDuration': [1000, 3000, 900],
            'appDuration_pred': [500, -1, 0],
            'speedup': [2.0, float('nan'), 0.0],
        })

        result = _calibrate_summary(summary.copy())

        assert result['speedup'].iloc[0] == pytest.approx(3.6)
        assert result['appDuration_pred'].iloc[0] == 278
        # the other two apps are left as they were
        assert pd.isna(result['speedup'].iloc[1]) and result['speedup'].iloc[2] == 0.0
        assert result['appDuration_pred'].tolist()[1:] == [-1, 0]
        assert pd.api.types.is_integer_dtype(result['appDuration_pred'])

        # no app has a usable speedup
        result = _calibrate_summary(summary.iloc[1:].copy())
        assert result['speedup'].tolist()[1] == 0.0
        assert result['appDuration_pred'].tolist() == [-1, 0]

    def test_summarize(self, monkeypatch, calibration):
        """Test that _summarize calibrates the app speedups, and adds the model id only when the config has one"""
        get_config(reload=True)
        config = types.SimpleNamespace(model_id=None)
        monkeypatch.setattr(f'{QUALX_MAIN}.get_config', lambda: config)
        monkeypatch.setattr(f'{QUALX_MAIN}.get_calibration', lambda: None)

        # SQLs are rolled up to their apps, and app-level rows are the apps
        for level, predictions, speedups in [('sql', sql_predictions(), [2.0, 2.0]),
                                             ('app', app_predictions(), [2.0, 1.5])]:
            summary = _summarize(predictions, level)
            assert summary['appId'].tolist() == ['id1', 'id2']
            assert summary['speedup'].tolist() == speedups
            assert summary['speedup_raw'].tolist() == speedups
            assert 'modelId' not in summary

        # with a model id and a calibration
        config.model_id = 'emr/yarn_cost/3'
        monkeypatch.setattr(f'{QUALX_MAIN}.get_calibration', lambda: calibration)
        for level, predictions, speedups, calibrated in [('sql', sql_predictions(), [2.0, 2.0], [3.6, 3.6]),
                                                         ('app', app_predictions(), [2.0, 1.5], [3.6, 2.6])]:
            summary = _summarize(predictions, level)
            assert summary['modelId'].tolist() == ['emr/yarn_cost/3', 'emr/yarn_cost/3']
            assert summary['speedup_raw'].tolist() == speedups
            assert summary['speedup'].tolist() == pytest.approx(calibrated)

    def test_predict_summary_with_calibration(self, prediction, tmp_path):
        """Test the per-app predictions of predict under a config with a calibration"""
        config_path = write_config(tmp_path / 'qualx-conf.yaml', CALIBRATION + 'model_id: emr/yarn_cost/3\n')

        per_app = predict('onprem', 'qual_output', prediction.output_info, config=config_path, qual_handlers=['qual'])

        assert per_app['appId'].tolist() == ['id1', 'id2', 'id3']
        # the predicted apps are calibrated, and keep the speedup that their SQLs rolled up to
        assert per_app['speedup_raw'].tolist() == [2.0, 2.0, 1.0]
        assert per_app['speedup'].tolist() == pytest.approx([3.6, 3.6, 1.0])
        # the app without a prediction keeps the default speedup, and its duration
        assert per_app['appDuration_pred'].tolist() == [56, 56, 500]
        assert per_app['modelId'].tolist() == ['emr/yarn_cost/3'] * 3

        written = pd.read_csv(prediction.output_info['perApp']['path'], index_col=0)
        assert written['speedup_raw'].tolist() == [2.0, 2.0, 1.0]
        assert written['speedup'].tolist() == pytest.approx([3.6, 3.6, 1.0])

    def test_predict_summary(self, prediction, tmp_path):
        """Test the per-app predictions of predict under a config with a model id"""
        config_path = write_config(tmp_path / 'qualx-conf.yaml', 'model_id: emr/yarn_cost/3\n')

        per_app = predict('onprem', 'qual_output', prediction.output_info, config=config_path, qual_handlers=['qual'])

        assert per_app['appId'].tolist() == ['id1', 'id2', 'id3']
        assert per_app['wasPredicted'].tolist() == [True, True, False]
        # the predicted apps have the speedup that their SQLs rolled up to
        assert per_app['speedup'].tolist() == [2.0, 2.0, 1.0]
        # the app without a prediction has the default speedup, and its duration
        assert per_app['appDuration_pred'].tolist() == [100, 100, 500]
        assert per_app['modelId'].tolist() == ['emr/yarn_cost/3'] * 3

        written = pd.read_csv(prediction.output_info['perApp']['path'], index_col=0)
        assert written['speedup'].tolist() == [2.0, 2.0, 1.0]
        assert written['modelId'].tolist() == ['emr/yarn_cost/3'] * 3

        # the raw features of the prediction are written, and those of the other level are not extracted
        assert prediction.calls == [('stage', 'sql')]
        written = pd.read_parquet(prediction.output_info['rawFeatures']['path'])
        pd.testing.assert_frame_equal(written, prediction.raw_features['sql'])
        assert not os.path.exists(prediction.output_info['rawFeaturesApp']['path'])

    def test_predict_without_config(self, prediction, tmp_path):
        """Test that predict without a config uses the default config, and not the config of an earlier call"""
        config_path = write_config(
            tmp_path / 'qualx-conf.yaml', 'model_id: emr/yarn_cost/3\nfeature_levels: [sql, app]\n')
        per_app = predict('onprem', 'qual_output', prediction.output_info, config=config_path, qual_handlers=['qual'])
        assert get_config().file_path == config_path
        assert 'modelId' in per_app
        # the config asks for the raw features of both levels
        assert prediction.calls == [('stage', 'sql'), ('stage', 'app')]
        for name, level in [('rawFeatures', 'sql'), ('rawFeaturesApp', 'app')]:
            written = pd.read_parquet(prediction.output_info[name]['path'])
            pd.testing.assert_frame_equal(written, prediction.raw_features[level])
            os.remove(prediction.output_info[name]['path'])
        prediction.calls.clear()

        per_app = predict('onprem', 'qual_output', prediction.output_info, qual_handlers=['qual'])

        assert get_config().file_path == default_config_path()
        assert _prediction_config(None) is get_config()
        assert 'modelId' not in per_app
        assert per_app['speedup'].tolist() == [2.0, 2.0, 1.0]
        assert prediction.calls == [('stage', 'sql')]
        assert os.path.exists(prediction.output_info['rawFeatures']['path'])
        assert not os.path.exists(prediction.output_info['rawFeaturesApp']['path'])

    def test_predict_at_app_level(self, prediction, tmp_path):
        """Test predict under a config that aggregates to the app before predicting"""
        config_path = write_config(
            tmp_path / 'qualx-conf.yaml', 'aggregation: app\nfeature_levels: [sql, app]\nmodel_id: emr/yarn_cost/3\n')
        args = {'config': config_path, 'qual_tool_filter': 'sqlId', 'qual_handlers': ['qual']}

        per_app = predict('onprem', 'qual_output', prediction.output_info, **args)

        # the app-level rows are predicted, and the SQL-level rows are extracted to be written
        assert prediction.calls == [('sqlId', 'app'), ('sqlId', 'sql')]
        assert per_app['appId'].tolist() == ['id1', 'id2', 'id3']
        assert per_app['wasPredicted'].tolist() == [True, True, False]
        assert per_app['speedup'].tolist() == [2.0, 1.5, 1.0]
        assert per_app['appDuration_pred'].tolist() == [500, 2000, 500]
        assert per_app['modelId'].tolist() == ['emr/yarn_cost/3'] * 3
        for name, level in [('rawFeatures', 'sql'), ('rawFeaturesApp', 'app')]:
            written = pd.read_parquet(prediction.output_info[name]['path'])
            pd.testing.assert_frame_equal(written, prediction.raw_features[level])
            os.remove(prediction.output_info[name]['path'])

        # the prediction does not need the rows of the other level
        prediction.failing_levels.add('sql')
        per_app = predict('onprem', 'qual_output', prediction.output_info, **args)
        assert per_app['speedup'].tolist() == [2.0, 1.5, 1.0]
        assert os.path.exists(prediction.output_info['rawFeaturesApp']['path'])
        assert not os.path.exists(prediction.output_info['rawFeatures']['path'])

        # and it needs the rows of its own level
        prediction.failing_levels.add('app')
        with pytest.raises(ValueError, match='No app-level features'):
            predict('onprem', 'qual_output', prediction.output_info, **args)

    def test_featurize(self, prediction):
        """Test that featurize extracts the raw features at the given level"""
        features = featurize('onprem', 'qual_output', qual_handlers=['qual'])
        assert features is prediction.raw_features['sql']
        features = featurize('onprem', 'qual_output', qual_handlers=['qual'], qual_tool_filter='sqlId', level='app')
        assert features is prediction.raw_features['app']
        assert prediction.calls == [('stage', 'sql'), ('sqlId', 'app')]

    def test_featurize_without_apps(self):
        """Test that featurize returns no rows when there are no apps to predict"""
        try:
            assert featurize('onprem', 'qual_output', qual_handlers=[]).empty
        finally:
            get_config(reload=True)  # reload config

    def test_predict_features(self, prediction, tmp_path):
        """Test that predict_features predicts raw features at the level of the config's aggregation"""
        predictions, summary = predict_features(prediction.raw_features['sql'], platform='onprem')
        assert predictions is prediction.predictions['sql']
        # the apps that have features
        assert summary['appId'].tolist() == ['id1', 'id2']
        assert summary['speedup'].tolist() == [2.0, 2.0]
        assert 'modelId' not in summary

        config_path = write_config(tmp_path / 'qualx-conf.yaml', 'aggregation: app\nmodel_id: emr/yarn_cost/3\n')
        predictions, summary = predict_features(
            prediction.raw_features['app'], platform='onprem', model='xgb_model.json', config=config_path)
        assert predictions is prediction.predictions['app']
        assert summary['appId'].tolist() == ['id1', 'id2']
        assert summary['speedup'].tolist() == [2.0, 1.5]
        assert summary['speedup_raw'].tolist() == [2.0, 1.5]
        assert summary['modelId'].tolist() == ['emr/yarn_cost/3', 'emr/yarn_cost/3']

        # with a calibration, the app speedups are calibrated and the raw ones are kept
        config_path = write_config(tmp_path / 'qualx-conf-calibrated.yaml', CALIBRATION + 'aggregation: app\n')
        _, summary = predict_features(
            prediction.raw_features['app'], platform='onprem', model='xgb_model.json', config=config_path)
        assert summary['speedup_raw'].tolist() == [2.0, 1.5]
        assert summary['speedup'].tolist() == pytest.approx([3.6, 2.6])

        # without a config, the default config
        _, summary = predict_features(prediction.raw_features['sql'], platform='onprem')
        assert summary['speedup'].tolist() == [2.0, 2.0]
        assert 'modelId' not in summary

    def test_predict_cli_passes_config(self, prediction, monkeypatch, tmp_path):
        """Test that _predict_cli forwards its config to predict rather than resetting to the default config"""
        config_path = write_config(tmp_path / 'qualx-conf.yaml', 'model_id: emr/yarn_cost/3\n')
        out_dir = tmp_path / 'out'
        monkeypatch.setattr(f'{QUALX_MAIN}.QualCore', lambda path: 'qual')

        _predict_cli('onprem', str(out_dir), qual_output='qual_output', config=config_path)

        assert get_config().file_path == config_path
        assert prediction.calls == [('stage', 'sql')]
        written = pd.read_csv(out_dir / 'per_app.csv', index_col=0)
        assert written['modelId'].tolist() == ['emr/yarn_cost/3'] * 3
