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

"""Test rollup module"""

import types

import numpy as np
import pandas as pd
import pytest  # pylint: disable=import-error

from spark_rapids_tools.tools.qualx.rollup import CostFunction
from ..conftest import SparkRapidsToolsUT

# each built-in label's total over its app
APP_TOTALS = {'Duration': 'appWallTime', 'duration_sum': 'appTaskTime'}


@pytest.fixture(name='sqls')
def _sqls():
    """One app with one supported SQL that holds 64,192 ms of its 74,094 ms, and all of its 3,000 ms of task time."""
    return pd.DataFrame({
        'appId': ['app1'],
        'Duration': [64192],
        'duration_sum': [3000],
        'appWallTime': [74094],
        'appTaskTime': [3000],
        'fraction_supported': [1.0],
        'y_pred': [2.808],
    })


@pytest.fixture(name='many_sqls')
def _many_sqls():
    """300 apps of 1 to 6 SQLs, some partly supported and some that add up to more than their app."""
    rng = np.random.default_rng(0)
    app_ids = np.repeat(np.arange(300), rng.integers(1, 7, 300))
    num_sqls = len(app_ids)
    sqls = pd.DataFrame({
        'appId': app_ids,
        'Duration': rng.integers(1, 100_000, num_sqls),
        'duration_sum': rng.integers(1, 1_000_000, num_sqls),
        'fraction_supported': rng.choice([1.0, 1.0, 0.0, 0.25, 0.9], num_sqls),
        'y_pred': rng.uniform(0.3, 8.0, num_sqls),
    })
    for label, app_total in APP_TOTALS.items():
        # an app's total is 0.5 to 2 times the sum of its SQLs
        in_sqls = sqls.groupby('appId')[label].transform('sum')
        sqls[app_total] = (in_sqls * sqls['appId'].map(pd.Series(rng.uniform(0.5, 2.0, 300)))).round()
    return sqls


def label_rollup(sqls: pd.DataFrame, label: str) -> pd.DataFrame:
    """Each app's duration, predicted duration, supported fraction and speedup as rolled up from the label, before
    cost functions."""
    supported = sqls[label] * sqls['fraction_supported']
    pred = sqls[label] * (1.0 - sqls['fraction_supported'] + sqls['fraction_supported'] / sqls['y_pred'])
    apps = sqls.assign(supported=supported, pred=pred).groupby('appId').agg(
        label=(label, 'sum'), supported=('supported', 'sum'), pred=('pred', 'sum'),
        app_duration=(APP_TOTALS[label], 'first'))
    app_duration = apps['app_duration'].clip(lower=apps['label'])
    app_duration_pred = app_duration - apps['label'] + apps['pred']
    return pd.DataFrame({
        'cost': app_duration,
        'gpuCost': app_duration_pred,
        'fraction_supported': apps['supported'] / app_duration,
        'speedup': app_duration / app_duration_pred,
    })


def make_plugin(**attrs) -> types.ModuleType:
    """A cost function of wall time, with attrs in place of its own."""
    plugin = types.ModuleType('wall_time')
    plugin.LABEL = 'Duration'
    plugin.INPUT_COLUMNS = ['Duration', 'appWallTime']
    plugin.sql_cost = lambda sqls: sqls['Duration']
    plugin.app_cost = lambda sqls: sqls['appWallTime']
    for name, value in attrs.items():
        setattr(plugin, name, value)
    return plugin


def write_plugin(tmp_path, name: str, source: str) -> str:
    path = tmp_path / name
    path.write_text(source, encoding='utf-8')
    return str(path)


class TestRollup(SparkRapidsToolsUT):
    """Test class for rollup module"""

    def test_duration(self, sqls):
        """The time outside the SQL keeps its wall time"""
        cost_fn = CostFunction.load('duration.py')
        app = cost_fn.rollup(sqls, ['appId']).iloc[0]
        assert cost_fn.label == 'Duration'
        assert app['cost'] == 74094
        assert app['gpuCost'] == pytest.approx(64192 / 2.808 + 9902)
        assert app['fraction_supported'] == pytest.approx(64192 / 74094)
        assert app['speedup'] == pytest.approx(2.262, abs=1e-3)

    def test_duration_sum(self, sqls):
        """All of the task time is in the SQL, so the app's speedup is the SQL's"""
        cost_fn = CostFunction.load('duration_sum.py')
        app = cost_fn.rollup(sqls, ['appId']).iloc[0]
        assert cost_fn.label == 'duration_sum'
        assert app['fraction_supported'] == 1.0
        assert app['speedup'] == pytest.approx(2.808)

    @pytest.mark.parametrize('label', ['Duration', 'duration_sum'])
    def test_time_costs_match_label_rollup(self, many_sqls, label):
        """The time costs roll up as their labels did, with unsupported parts and overlapping SQLs"""
        apps = CostFunction.load(f'{label}.py'.lower()).rollup(many_sqls, ['appId']).set_index('appId')
        pd.testing.assert_frame_equal(apps, label_rollup(many_sqls, label), check_dtype=False, rtol=1e-12)

    def test_label(self, sqls):
        """The label is the cost function's own cost of a SQL, or that of the built-in that LABEL names"""
        runs = pd.concat([sqls.assign(runType='CPU'), sqls.assign(runType='GPU', duration_sum=1000)])
        # task time, of each SQL in its own run
        duration_sum = CostFunction.load('duration_sum.py')
        assert duration_sum.with_label(runs)['duration_sum'].tolist() == [3000, 1000]
        assert duration_sum.app_cost(runs).tolist() == [3000, 3000]
        # a plugin with a label of its own: the model learns the ratio of its cost
        own = CostFunction(make_plugin(LABEL='halfDuration', sql_cost=lambda sqls: sqls['Duration'] / 2))
        assert own.label == 'halfDuration'
        assert own.with_label(runs)['halfDuration'].tolist() == [32096, 32096]
        # a plugin that names the label of a built-in reads the built-in's model, of task time here
        named = CostFunction(make_plugin(LABEL='duration_sum', sql_cost=lambda sqls: sqls['Duration'] / 2))
        assert named.with_label(runs)['duration_sum'].tolist() == [3000, 1000]
        assert named.sql_cost(runs).tolist() == [32096, 32096]

    def test_gpu_cost_frames(self):
        """gpu_sql_cost gets each SQL's cost and adjusted ratio, and gpu_app_cost gets each SQL with its GPU cost"""
        sqls = pd.DataFrame({
            'appId': ['app1', 'app1', 'app2'],
            'sqlID': [0, 1, 0],
            'Duration': [100, 300, 50],
            'appWallTime': [500, 500, 40],
            'fraction_supported': [1.0, 0.25, np.nan],
            'y_pred': [2.0, 4.0, 3.0],
        })
        seen = {}

        def gpu_sql_cost(sqls, speedup):
            seen['speedup'] = speedup
            return sqls['cost'] / speedup

        def gpu_app_cost(sqls):
            seen['sqls'] = sqls
            # the time outside an app's SQLs takes twice as long on GPU
            app = sqls.groupby('app')
            return app['gpuCost'].transform('sum') + 2 * (sqls['appCost'] - app['cost'].transform('sum')).clip(lower=0)

        cost_fn = CostFunction(make_plugin(gpu_sql_cost=gpu_sql_cost, gpu_app_cost=gpu_app_cost))
        apps = cost_fn.rollup(sqls, ['appId'])
        # the unsupported part of a SQL has a ratio of 1, and a SQL with no fraction is unsupported
        assert seen['speedup'].tolist() == pytest.approx([2.0, 1 / (0.75 + 0.25 / 4.0), 1.0])
        # each SQL has its cost, its app's cost, its GPU cost and its app
        assert seen['sqls']['cost'].tolist() == [100, 300, 50]
        assert seen['sqls']['appCost'].tolist() == [500, 500, 40]
        assert seen['sqls']['gpuCost'].tolist() == pytest.approx([100 / 2.0, 300 * (0.75 + 0.25 / 4.0), 50.0])
        assert seen['sqls']['app'].tolist() == [0, 0, 1]
        # app1 has 100 ms outside its SQLs, and app2's SQL is longer than its app
        assert apps['cost'].tolist() == [500, 50]
        assert apps['gpuCost'].tolist() == pytest.approx([100 / 2.0 + 300 * (0.75 + 0.25 / 4.0) + 2 * 100, 50.0])
        assert apps['fraction_supported'].tolist() == pytest.approx([(100 + 75) / 500, 0.0])

    def test_price_ratio(self, sqls):
        """The prices scale the speedup by their ratio. They aren't in the costs, or in the label of training"""
        unpriced = CostFunction.load('duration.py').rollup(sqls, ['appId']).iloc[0]
        # a CPU cluster at 14.60 an hour, and a GPU cluster at 15.96
        price_ratio = 14.60 / 15.96
        cost_fn = CostFunction.load('duration.py', price_ratio=price_ratio)
        priced = cost_fn.rollup(sqls, ['appId']).iloc[0]
        assert priced['speedup'] == pytest.approx(unpriced['speedup'] * price_ratio)
        assert (priced['cost'], priced['gpuCost']) == (unpriced['cost'], unpriced['gpuCost'])
        assert cost_fn.with_label(sqls)['Duration'].tolist() == [64192]

    def test_sql_without_fraction_is_unsupported(self, sqls):
        """A SQL with no task time has no supported fraction, and keeps its cost"""
        cost_fn = CostFunction.load('duration.py')
        app = cost_fn.rollup(sqls.assign(fraction_supported=np.nan), ['appId']).iloc[0]
        assert app['speedup'] == 1.0

    def test_args(self):
        """Loading fails on a missing or an unknown arg"""
        plugin = make_plugin(gpu_sql_cost=lambda sqls, speedup, scale: sqls['cost'] / speedup * scale)
        with pytest.raises(ValueError, match='missing args for gpu_sql_cost'):
            CostFunction(plugin)
        with pytest.raises(ValueError, match='takes no args'):
            CostFunction(plugin, {'scale': 2.0, 'shift': 1.0})
        with pytest.raises(ValueError, match='takes no args'):
            CostFunction.load('duration.py', {'scale': 2.0})

    def test_plugin(self, tmp_path, sqls):
        """A plugin outside of qualx sets its own cost for the time outside SQLs"""
        path = write_plugin(tmp_path, 'slow_non_sql.py', '''
LABEL = 'Duration'
INPUT_COLUMNS = ['Duration', 'appWallTime', 'numExecutors']


def sql_cost(sqls):
    return sqls['Duration']


def app_cost(sqls):
    return sqls['appWallTime']


def gpu_app_cost(sqls, non_sql_speedup=1.0):
    app = sqls.groupby('app')
    non_sql_cost = (sqls['appCost'] - app['cost'].transform('sum')).clip(lower=0)
    return app['gpuCost'].transform('sum') + non_sql_cost / non_sql_speedup
''')
        cost_fn = CostFunction.load(path, {'non_sql_speedup': 0.5})
        assert cost_fn.input_columns == ['Duration', 'appWallTime', 'numExecutors']
        with pytest.raises(ValueError, match=r"missing columns: \['numExecutors'\]"):
            cost_fn.rollup(sqls, ['appId'])
        sqls = sqls.assign(numExecutors=37)
        app = cost_fn.rollup(sqls, ['appId']).iloc[0]
        assert app['gpuCost'] == pytest.approx(64192 / 2.808 + 9902 / 0.5)
        # the arg has a default
        assert CostFunction.load(path).rollup(sqls, ['appId'])['gpuCost'].iloc[0] == pytest.approx(64192 / 2.808 + 9902)

    def test_invalid_plugin(self, tmp_path):
        """A plugin defines its LABEL, its INPUT_COLUMNS and the cost of a SQL and of an app"""
        with pytest.raises(ValueError, match=r"does not define: \['INPUT_COLUMNS', 'sql_cost', 'app_cost'\]"):
            CostFunction.load(write_plugin(tmp_path, 'no_functions.py', "LABEL = 'Duration'\n"))

    def test_missing_columns(self, sqls):
        """The costs fail on features without the cost function's columns, as cached before they existed"""
        with pytest.raises(ValueError, match=r"missing columns: \['appWallTime'\]"):
            CostFunction.load('duration.py').rollup(sqls.drop(columns='appWallTime'), ['appId'])
        with pytest.raises(ValueError, match=r"missing columns: \['appTaskTime'\]"):
            CostFunction.load('duration_sum.py').app_cost(sqls.drop(columns='appTaskTime'))

    def test_different_app_costs(self, sqls):
        """The rollup fails if the SQLs of one app don't agree on its cost, or on its GPU cost"""
        two_sqls = pd.concat([sqls, sqls.assign(appWallTime=80000)])
        with pytest.raises(ValueError, match='different values of app_cost'):
            CostFunction.load('duration.py').rollup(two_sqls, ['appId'])
        # a gpu_app_cost that gives each SQL its own GPU cost, and not its app's
        cost_fn = CostFunction(make_plugin(gpu_app_cost=lambda sqls: sqls['gpuCost']))
        with pytest.raises(ValueError, match='different values of gpu_app_cost'):
            cost_fn.rollup(pd.concat([sqls, sqls.assign(Duration=1000)]), ['appId'])

    def test_invalid_costs(self, sqls):
        """The rollup fails on a cost that is missing or negative, and not on an app with no cost"""
        cost_fn = CostFunction.load('duration_sum.py')
        with pytest.raises(ValueError, match=r'SQL\(s\) a missing or negative cost'):
            cost_fn.rollup(sqls.assign(duration_sum=-1), ['appId'])
        with pytest.raises(ValueError, match=r'SQL\(s\) a missing or negative cost'):
            cost_fn.rollup(sqls.assign(duration_sum=np.nan), ['appId'])
        with pytest.raises(ValueError, match=r'app\(s\) a missing or negative cost'):
            cost_fn.rollup(sqls.assign(appTaskTime=np.nan), ['appId'])
        with pytest.raises(ValueError, match=r'SQL\(s\) a missing or negative gpuCost'):
            cost_fn.rollup(sqls.assign(y_pred=-1.0), ['appId'])
        negative_gpu_app_cost = CostFunction(make_plugin(gpu_app_cost=lambda sqls: -sqls['gpuCost']))
        with pytest.raises(ValueError, match=r'app\(s\) a missing or negative gpuCost'):
            negative_gpu_app_cost.rollup(sqls, ['appId'])
        app = cost_fn.rollup(sqls.assign(duration_sum=0, appTaskTime=0), ['appId']).iloc[0]
        assert app['cost'] == 0
        assert np.isnan(app['speedup'])
