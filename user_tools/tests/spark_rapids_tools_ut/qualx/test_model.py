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

"""Test qualx_model module"""

import random

import pytest  # pylint: disable=import-error
import pandas as pd
import numpy as np

from xgboost import Booster

from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.preprocess import expected_raw_features
from spark_rapids_tools.tools.qualx.model import (
    compute_sample_weights,
    extract_model_features,
    predict,
    train,
    calibrate,
    compute_shapley_values,
)
from spark_rapids_tools.tools.qualx.util import get_abs_path, load_plugin
from ..conftest import SparkRapidsToolsUT


class TestModel(SparkRapidsToolsUT):
    """Test class for qualx_model module"""
    @staticmethod
    def generate_test_data(label_col: str, seed: int = 0) -> pd.DataFrame:
        # Create test data with 200 rows (100 CPU/GPU pairs)
        np.random.seed(seed)
        random.seed(seed)

        # fill in random features and random labels
        fixed_features = set(['appName', 'description', 'runType', 'scaleFactor', 'sqlID'])
        random_features = expected_raw_features() - fixed_features
        features = {}
        for feature in random_features:
            features[feature] = np.random.rand(200)
        df = pd.DataFrame(features)
        df[label_col] = np.random.rand(200)

        # fill in fixed features
        df['appName'] = 'test_app'
        df['description'] = 'testing'
        df['scaleFactor'] = 1
        df.loc[0:99, 'sqlID'] = range(100)
        df.loc[0:99, 'runType'] = 'CPU'
        df.loc[100:199, 'sqlID'] = range(100)
        df.loc[100:199, 'runType'] = 'GPU'

        return df

    @pytest.mark.parametrize('label', ['Duration', 'duration_sum'])
    def test_train_and_predict(self, monkeypatch, label) -> None:
        # mock os.environ to override the default label
        monkeypatch.setenv('QUALX_LABEL', label)
        get_config(reload=True)  # reload config

        # Create test data
        df = self.generate_test_data(label)

        # Extract features
        plugin = load_plugin(get_abs_path('split_random.py', 'split_functions'))
        features, feature_cols, label_col = extract_model_features(
            df, split_functions={'default': plugin.split_function}
        )

        # Assert label_col column exists
        assert label_col == f'{label}_speedup'
        assert label_col in features.columns

        # Assert label column is in features, but not in list of feature_cols
        assert label in features.columns
        assert label not in feature_cols

        model = train(features, feature_cols, label_col, n_trials=5)

        # Assert model is trained
        assert isinstance(model, Booster)
        assert model.num_boosted_rounds() > 0

        # Make predictions
        results = predict(model, features, feature_cols, label_col)

        # Assert predictions are made
        assert 'y_pred' in results.columns
        assert f'{label}_pred' in results.columns
        assert len(results) == 100

        # Compute SHAP values
        feature_importance, shap_values = compute_shapley_values(model, features)

        # Assert results
        assert isinstance(feature_importance, pd.DataFrame)
        assert isinstance(shap_values, pd.DataFrame)
        assert 'feature' in feature_importance.columns
        assert 'shap_value' in feature_importance.columns
        assert len(shap_values.columns) == len(feature_cols) + 1

        # Assert the model records its label, and prediction fails under a cost function with another label
        assert model.attr('qualx_label') == label
        model.set_attr(qualx_label='duration_sum' if label == 'Duration' else 'Duration')
        with pytest.raises(ValueError, match='Model was trained on the label'):
            predict(model, features, feature_cols, label_col)

    def test_compute_weights(self) -> None:
        """Test compute_weights function"""
        # Create test data
        df = self.generate_test_data('Duration')  # Use default label
        features, _, label_col = extract_model_features(df)
        y = features[label_col]

        threshold = 1.0
        num_positives = y[y > threshold].count()
        num_negatives = y[y <= threshold].count()

        # Test with manual weights
        positive_weight, negative_weight = compute_sample_weights(y, threshold, 2.0, 3.0)
        assert positive_weight == 2.0
        assert negative_weight == 3.0

        # Test with auto weights
        positive_weight, negative_weight = compute_sample_weights(y, threshold, 'auto', 'auto')
        assert positive_weight == np.max([num_negatives / num_positives, 1.0])
        assert negative_weight == np.max([num_positives / num_negatives, 1.0])

        positive_weight, negative_weight = compute_sample_weights(y, threshold, 2.0, 'auto')
        assert positive_weight == 2.0
        assert negative_weight == np.max([num_positives / num_negatives, 1.0])

        positive_weight, negative_weight = compute_sample_weights(y, threshold, 'auto', 2.0)
        assert positive_weight == np.max([num_negatives / num_positives, 1.0])
        assert negative_weight == 2.0

        # Test with auto weights and unbalanced samples
        threshold = 2.0
        num_positives = y[y > threshold].count()
        num_negatives = y[y <= threshold].count()

        positive_weight, negative_weight = compute_sample_weights(y, threshold, 'auto', 'auto')
        assert positive_weight == np.max([num_negatives / num_positives, 1.0])
        assert negative_weight == np.max([num_positives / num_negatives, 1.0])

        # Test with auto weights and zero positives
        threshold = y.max() + 1.0
        num_positives = y[y > threshold].count()
        num_negatives = y[y <= threshold].count()
        assert num_positives == 0
        assert num_negatives == 100

        positive_weight, negative_weight = compute_sample_weights(y, threshold, 'auto', 'auto')
        assert positive_weight == 1.0
        assert negative_weight == 1.0

        # Test with auto weights and zero negatives
        threshold = y.min() - 1.0
        num_positives = y[y > threshold].count()
        num_negatives = y[y <= threshold].count()
        assert num_positives == 100
        assert num_negatives == 0

        positive_weight, negative_weight = compute_sample_weights(y, threshold, 'auto', 'auto')
        assert positive_weight == 1.0
        assert negative_weight == 1.0

    def test_train_with_sample_weight(self, monkeypatch) -> None:
        """Test training models with different sample weights"""
        # Create test data
        df = self.generate_test_data('Duration')  # Use default label
        plugin = load_plugin(get_abs_path('split_random.py', 'split_functions'))
        features, feature_cols, label_col = extract_model_features(
            df, split_functions={'default': plugin.split_function}
        )

        # Mock config for default weights
        default_config = get_config(reload=True)
        default_config.sample_weight = {}
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: default_config)

        # Train model with default weights
        default_model = train(features, feature_cols, label_col, n_trials=5)

        # Make predictions with default model
        default_results = predict(default_model, features, feature_cols, label_col)

        # Mock config for custom weights
        weighted_config = get_config(reload=True)
        weighted_config.sample_weight = {
            'threshold': 1.0,               # speedup threshold
            'positive': 2.0,                # weight for speedup > threshold
            'negative': 1.0                 # weight for speedup <= threshold
        }
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: weighted_config)

        # Train model with custom weights
        weighted_model = train(features, feature_cols, label_col, n_trials=5)

        # Make predictions with weighted model
        weighted_results = predict(weighted_model, features, feature_cols, label_col)

        # The weighted model should generally predict higher values for positive samples
        positive_mask = features[label_col] > 1.0
        positive_diff = weighted_results.loc[positive_mask, 'y_pred'] - default_results.loc[positive_mask, 'y_pred']
        assert positive_diff.mean() > 0, 'Weighted model should predict higher values for positive samples'

    def test_train_with_calib(self, monkeypatch) -> None:
        """Test training models with and without calibration"""
        # Create test data
        df = self.generate_test_data('Duration')  # Use default label
        plugin = load_plugin(get_abs_path('split_random.py', 'split_functions'))
        features, feature_cols, label_col = extract_model_features(
            df, split_functions={'default': plugin.split_function}
        )

        # Mock config for default config without calibration
        default_config = get_config(reload=True)
        default_config.calib = False
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: default_config)

        # Train model with default config without calibration
        default_model = train(features, feature_cols, label_col, n_trials=5)

        # Make predictions with default model without calibration
        default_results = predict(default_model, features, feature_cols, label_col)

        # Mock config with calibration enabled
        config_with_calib = get_config(reload=True)
        config_with_calib.calib = True
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: config_with_calib)

        # Train model with custom weights
        precalib_model = train(features, feature_cols, label_col, n_trials=5)
        calib_params = calibrate(features, feature_cols, label_col, precalib_model)

        # Make predictions with weighted model
        calib_results = predict(precalib_model, features, feature_cols, label_col, calib_params)

        # The calibrated model should generally predict higher values for positive samples
        positive_mask = features[label_col] > features[label_col].mean()
        positive_diff = calib_results.loc[positive_mask, 'y_pred'] - default_results.loc[positive_mask, 'y_pred']
        assert positive_diff.mean() > 0, 'Calib model should predict higher values for positive samples'
