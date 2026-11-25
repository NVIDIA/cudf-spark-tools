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

"""Test qualx_config module"""
import os
import tempfile

import pytest  # pylint: disable=import-error
from pydantic import ValidationError

from spark_rapids_tools.tools.qualx.qualx_config import QualxConfig, QualxPipelineConfig
from ..conftest import SparkRapidsToolsUT


@pytest.fixture(name='qualx_config_params')
def _qualx_config_params():
    """Fixture providing common parameters for QualxConfig"""
    return {
        'datasets': 'datasets',
        'featurizers': ['default.py'],
        'modifiers': ['align_hash.py'],
        'split_functions': {'train': 'plugins/split_stratified.py', 'test': 'plugins/split_all_test.py'},
        'model_type': 'xgboost',
        'xgboost': {'model_name': 'xgb_model.json', 'n_trials': 200, 'qual_tool_filter': 'stage'}
    }


@pytest.fixture(name='qualx_pipeline_config_params')
def _qualx_pipeline_config_params(qualx_config_params):
    """Fixture providing common parameters for QualxPipelineConfig"""
    return {
        'dataset_name': 'test_dataset',
        'platform': 'onprem',
        'eventlogs': {'cpu': ['/path/to/cpu/eventlogs'], 'gpu': ['/path/to/gpu/eventlogs']},
        'output_dir': 'test_output',
        'alignment_dir': 'alignment',
        **qualx_config_params
    }


class TestQualxConfig(SparkRapidsToolsUT):
    """Test class for QualxConfig"""

    def test_default_values(self, qualx_config_params):
        """Test that QualxConfig has correct default values"""
        config = QualxConfig(**qualx_config_params)
        assert config.cache_dir == 'qualx_cache'
        assert config.label == 'Duration'
        assert config.alignment_dir is None
        assert config.aggregation == 'sql'
        assert config.feature_levels is None
        assert config.model_id is None
        assert config.calibration is None

    def test_aggregation(self, qualx_config_params):
        """Test that aggregation is sql or app"""
        for aggregation in ['sql', 'app']:
            assert QualxConfig(**qualx_config_params, aggregation=aggregation).aggregation == aggregation
        for invalid in ['stage', 'App', None]:
            with pytest.raises(ValidationError):
                QualxConfig(**qualx_config_params, aggregation=invalid)

    def test_feature_levels(self, qualx_config_params):
        """Test that feature_levels is a list of sql and app"""
        for levels in [[], ['sql'], ['app'], ['sql', 'app'], ['app', 'sql']]:
            assert QualxConfig(**qualx_config_params, feature_levels=levels).feature_levels == levels
            assert QualxConfig(**qualx_config_params, aggregation='app', feature_levels=levels).feature_levels == levels
        for invalid in [['stage'], ['sql', 'job'], 'app']:
            with pytest.raises(ValidationError):
                QualxConfig(**qualx_config_params, feature_levels=invalid)

    def test_model_id(self, qualx_config_params):
        """Test that model_id is a string"""
        config = QualxConfig(**qualx_config_params, model_id='emr/yarn_cost/3')
        assert config.model_id == 'emr/yarn_cost/3'
        with pytest.raises(ValidationError):
            QualxConfig(**qualx_config_params, model_id=['emr', 3])

    def test_calibration(self, qualx_config_params):
        """Test that calibration is a path with optional args"""
        assert QualxConfig(**qualx_config_params).calibration is None
        args = {'c0': 0.51, 'c1': 1.0, 'c2': 2.52, 'direction': 'inverse', 'floor': 0.5}
        for calibration in [{'path': 'power.py'}, {'path': 'power.py', 'args': args}, {'path': 'x.py', 'args': None}]:
            assert QualxConfig(**qualx_config_params, calibration=calibration).calibration == calibration

        for invalid in [{}, {'args': args}, {'path': 'power.py', 'level': 'app'}]:
            with pytest.raises(ValidationError, match='calibration takes a path and optional args'):
                QualxConfig(**qualx_config_params, calibration=invalid)
        for invalid in [{'path': 3}, {'path': 'power.py', 'args': [0.5, 1.0]}]:
            with pytest.raises(ValidationError, match='calibration takes a path and a dictionary of args'):
                QualxConfig(**qualx_config_params, calibration=invalid)
        with pytest.raises(ValidationError):
            QualxConfig(**qualx_config_params, calibration='power.py')

    def test_load_prediction_settings_from_file(self, tmp_path):
        """Test loading the settings of a prediction from a file"""
        config_path = tmp_path / 'qualx-conf.yaml'
        config_path.write_text("""
datasets: datasets
featurizers: ['default.py', 'hash_plan.py']
split_functions: split_train_val.py
model_type: xgboost
xgboost: {}
aggregation: app
feature_levels: [sql, app]
model_id: emr/yarn_cost/3
""", encoding='utf-8')
        config = QualxConfig.load_from_file(str(config_path))
        assert config.aggregation == 'app'
        assert config.feature_levels == ['sql', 'app']
        assert config.model_id == 'emr/yarn_cost/3'

    def test_environment_variable_overrides(self, monkeypatch, qualx_config_params):
        """Test that environment variables override default values"""
        # Test QUALX_CACHE_DIR override
        monkeypatch.setenv('QUALX_CACHE_DIR', 'custom_cache')
        config = QualxConfig(**qualx_config_params)
        assert config.cache_dir == 'custom_cache'

        # Test QUALX_LABEL override with valid value
        monkeypatch.setenv('QUALX_LABEL', 'duration_sum')
        config = QualxConfig(**qualx_config_params)
        assert config.label == 'duration_sum'

        # Test QUALX_LABEL override with invalid value
        monkeypatch.setenv('QUALX_LABEL', 'invalid_label')
        with pytest.raises(ValueError):
            QualxConfig(**qualx_config_params)

        # Test that environment variables are properly reset
        monkeypatch.delenv('QUALX_CACHE_DIR')
        monkeypatch.delenv('QUALX_LABEL')
        config = QualxConfig(**qualx_config_params)
        assert config.cache_dir == 'qualx_cache'
        assert config.label == 'Duration'

    def test_load_from_file(self):
        """Test loading configuration from a file"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write("""
cache_dir: test_cache
datasets: test_datasets
featurizers: ['featurizer1.py', 'featurizer2.py']
modifiers: ['modifier1.py']
label: duration_sum
split_functions:
  train: plugins/split_train.py
  test: plugins/split_test.py
model_type: xgboost
xgboost:
  model_name: test_model.json
  n_trials: 100
  qual_tool_filter: stage
""")
            temp_file_path = temp_file.name

        try:
            # Load the configuration from the file
            config = QualxConfig.load_from_file(temp_file_path)

            # Verify the loaded values
            assert config.cache_dir == 'test_cache'
            assert config.datasets == 'test_datasets'
            assert config.featurizers == ['featurizer1.py', 'featurizer2.py']
            assert config.modifiers == ['modifier1.py']
            assert config.label == 'duration_sum'
            assert config.split_functions == {
                'train': 'plugins/split_train.py',
                'test': 'plugins/split_test.py'
            }
            assert config.model_type == 'xgboost'
            assert config.xgboost == {
                'model_name': 'test_model.json',
                'n_trials': 100,
                'qual_tool_filter': 'stage'
            }
            assert config.file_path == temp_file_path
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    def test_get_schema(self):
        """Test that get_schema returns a valid JSON schema"""
        schema = QualxConfig.get_schema()
        assert isinstance(schema, dict)
        assert 'properties' in schema
        assert 'cache_dir' in schema['properties']
        assert 'label' in schema['properties']
        assert 'datasets' in schema['properties']

    def test_split_functions_as_dict(self, qualx_config_params):
        """Test that split_functions can be a dictionary with path and args"""
        # Create a dictionary version of split_functions
        split_dict = {
            'train': {
                'path': 'split_stratified.py',
                'args': {'threshold': 2.0, 'test_pct': 0.2, 'val_pct': 0.2}
            },
            'test': 'split_all_test.py'
        }

        # Update the config params with the dictionary version
        config_params = qualx_config_params.copy()
        config_params['split_functions'] = split_dict

        # Create config and verify the values
        config = QualxConfig(**config_params)
        assert config.split_functions == split_dict
        assert isinstance(config.split_functions, dict)
        assert 'train' in config.split_functions
        assert 'test' in config.split_functions
        assert config.split_functions['train']['path'] == 'split_stratified.py'
        assert config.split_functions['train']['args']['threshold'] == 2.0
        assert config.split_functions['test'] == 'split_all_test.py'


class TestQualxPipelineConfig(SparkRapidsToolsUT):
    """Test class for QualxPipelineConfig"""

    def test_default_values(self):
        """Test that QualxPipelineConfig has correct default values"""
        # QualxPipelineConfig requires dataset_name, platform, and eventlogs
        with pytest.raises(ValidationError):
            QualxPipelineConfig()

    def test_required_fields(self, qualx_pipeline_config_params):
        """Test that QualxPipelineConfig requires the necessary fields"""
        config = QualxPipelineConfig(**qualx_pipeline_config_params)

        assert config.dataset_name == 'test_dataset'
        assert config.platform == 'onprem'
        assert config.eventlogs == {
            'cpu': ['/path/to/cpu/eventlogs'],
            'gpu': ['/path/to/gpu/eventlogs']
        }
        assert config.output_dir == 'test_output'
        assert config.alignment_dir == 'alignment'

    def test_inheritance(self, qualx_pipeline_config_params):
        """Test that QualxPipelineConfig inherits from QualxConfig"""
        # Add QualxConfig specific overrides
        config_params = qualx_pipeline_config_params.copy()
        config_params['cache_dir'] = 'custom_cache'
        config_params['label'] = 'duration_sum'

        config = QualxPipelineConfig(**config_params)

        # Check inherited properties
        assert config.cache_dir == 'custom_cache'
        assert config.label == 'duration_sum'

        # Check QualxPipelineConfig specific properties
        assert config.dataset_name == 'test_dataset'
        assert config.platform == 'onprem'
        assert config.eventlogs == {
            'cpu': ['/path/to/cpu/eventlogs'],
            'gpu': ['/path/to/gpu/eventlogs']
        }
        assert config.output_dir == 'test_output'
        assert config.alignment_dir == 'alignment'
