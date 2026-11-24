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

"""Test qualx_config module"""
import pandas as pd
import pytest  # pylint: disable=import-error
from spark_rapids_tools.tools.qualx.config import (
    get_config,
    get_cache_dir,
    get_cost_function,
    get_label,
)
from ..conftest import SparkRapidsToolsUT


class TestConfig(SparkRapidsToolsUT):
    """Test class for qualx_config module"""
    def test_get_cache_dir(self, monkeypatch):
        # Test with mock environment variable
        monkeypatch.setenv('QUALX_CACHE_DIR', 'test_cache')
        assert get_cache_dir() == 'test_cache'

        # Test without environment variable (should use default)
        monkeypatch.delenv('QUALX_CACHE_DIR')
        get_config(reload=True)  # reload config
        assert get_cache_dir() == 'qualx_cache'

    def test_get_label(self, monkeypatch):
        # Test with duration_sum
        monkeypatch.setenv('QUALX_LABEL', 'duration_sum')
        get_config(reload=True)  # reload config
        assert get_label() == 'duration_sum'

        # Test with unsupported label
        with pytest.raises(ValueError):
            monkeypatch.setenv('QUALX_LABEL', 'duration')
            get_config(reload=True)  # reload config
            get_label()

        # Test without environment variable (should use default)
        monkeypatch.delenv('QUALX_LABEL')
        get_config(reload=True)  # reload config
        assert get_label() == 'Duration'

    def test_get_cost_function(self, tmp_path):
        # Test with a cost function next to the config, its args and the prices in the config
        (tmp_path / 'non_sql_speedup.py').write_text("""
LABEL = 'Duration'
INPUT_COLUMNS = ['Duration', 'appWallTime']


def sql_cost(sqls):
    return sqls['Duration']


def app_cost(sqls):
    return sqls['appWallTime']


def gpu_app_cost(sqls, non_sql_speedup):
    app = sqls.groupby('app')
    non_sql_cost = (sqls['appCost'] - app['cost'].transform('sum')).clip(lower=0)
    return app['gpuCost'].transform('sum') + non_sql_cost / non_sql_speedup
""", encoding='utf-8')
        config_path = tmp_path / 'qualx-conf.yaml'
        config_path.write_text("""
datasets: datasets
featurizers: ['default.py']
cost_function:
  path: non_sql_speedup.py
  args:
    non_sql_speedup: 0.5
cpu_cost: 0.5
gpu_cost: 2.0
split_functions: split_train_val.py
model_type: xgboost
xgboost: {}
""", encoding='utf-8')
        try:
            get_config(str(config_path), reload=True)
            # the app spent an hour in its SQL, and an hour outside it
            sqls = pd.DataFrame({'appId': ['app1'], 'Duration': 3_600_000, 'appWallTime': 7_200_000,
                                 'fraction_supported': 1.0, 'y_pred': 2.0})
            app = get_cost_function().rollup(sqls, ['appId']).iloc[0]
            assert app['cost'] == 7_200_000
            # with the args, the hour outside the SQL takes two on GPU
            assert app['gpuCost'] == (0.5 + 2) * 3_600_000
            # a unit of cost is a quarter of the price in a CPU run, which scales the speedup and not the costs
            assert app['speedup'] == pytest.approx(0.25 * 2 / 2.5)
            # the model learns the ratio of wall time
            assert get_label() == 'Duration'
        finally:
            get_config(reload=True)  # reload config

        # Test without a cost function or prices in the config (should use default)
        assert get_cost_function().sql_cost(sqls).tolist() == [3_600_000]
        assert get_cost_function().price_ratio == 1.0
