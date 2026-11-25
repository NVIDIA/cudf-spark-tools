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

"""Test the default featurizer"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest  # pylint: disable=import-error

from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.util import get_abs_path, load_plugin
from ..conftest import SparkRapidsToolsUT

# the task metrics that the profiler sums over the tasks of a job or a stage
SUM_METRICS = [
    'duration',
    'diskBytesSpilled',
    'memoryBytesSpilled',
    'input_bytesRead',
    'output_bytesWritten',
    'sr_localBytesRead',
    'sr_remoteBytesRead',
    'sr_remoteBytesReadToDisk',
    'sr_totalBytesRead',
    'sw_bytesWritten',
    'executorCPUTime',
    'executorDeserializeCPUTime',
    'executorDeserializeTime',
    'executorRunTime',
    'jvmGCTime',
    'sr_fetchWaitTime',
    'sw_writeTime',
    'input_recordsRead',
    'output_recordsWritten',
    'resultSerializationTime',
    'sr_localBlocksFetched',
    'sr_remoteBlocksFetched',
    'sw_recordsWritten',
]

# the attributes of an app, which are the same in each of its rows
APP_ATTRIBUTES = {
    'appDuration': 500000,
    'startTime': 1,
    'sparkRuntime': 'SPARK',
    'sparkVersion': '3.5.0',
    'pluginEnabled': False,
    'resourceProfileId': 0,
    'numExecutors': 8,
    'executorCores': 16,
    'maxMem': 1,
    'maxOnHeapMem': 1,
    'maxOffHeapMem': 0,
    'executorMemory': 64,
    'numGpusPerExecutor': 0,
    'executorOffHeap': 0,
    'taskCpu': 1.0,
    'taskGpu': 0.0,
}

# the operators of the nodes of each SQL of an app: a scan is named with its table, and a SQL can repeat an operator
SQL_OPS = [
    ['Scan parquet db.sales', 'Filter', 'Project'],
    ['Scan parquet db.items', 'Filter', 'Filter', 'Exchange', 'SortMergeJoin'],
    ['HashAggregate', 'Exchange', 'HashAggregate'],
    ['Scan parquet db.sales', 'Project', 'Exchange'],
]
# the number of the SQLs above that have each operator
SQL_OPS_COUNTS = {
    'sqlOp_Scan parquet': 3,
    'sqlOp_Filter': 2,
    'sqlOp_Project': 2,
    'sqlOp_Exchange': 3,
    'sqlOp_SortMergeJoin': 1,
    'sqlOp_HashAggregate': 1,
}


def make_app(
    rng: np.random.Generator,
    app_id: str,
    sql_ops: Sequence[Sequence[str]],
    *,
    with_cache: bool = False,
    with_support: bool = False,
    zero_metrics: Sequence[str] = (),
) -> Dict[str, pd.DataFrame]:
    """The tables that the featurizer loads for an app: its SQLs, their jobs, operators, data sources and failed
    tasks.

    The app has one SQL for each list of operators in sql_ops. It also has jobs that belong to no SQL, with the
    sqlID -1, and the jobs of one more SQL that is not in the app table, as when the loader drops a failed SQL.
    with_cache adds the cache metrics of a Databricks app to each SQL. with_support marks each row of task metrics
    as supported or not, as when the rows are the stages of a SQL. zero_metrics are summed metrics that are zero
    in every job.
    """
    num_sqls = len(sql_ops)
    jobs, ops, data_sources, failed_tasks, sqls = [], [], [], [], []
    for sql_id in [-1] + list(range(num_sqls + 1)):
        for job_index in range(int(rng.integers(2, 5))):
            job = {
                'appId': app_id,
                'appName': 'app',
                'sqlID': sql_id,
                'hasSqlID': sql_id != -1,
                'numTasks_sum': int(rng.integers(1, 400)),
                'jobStartTime_min': int(rng.integers(1, 10**6)),
                'duration_min': float(rng.integers(1, 50)),
                'duration_max': float(rng.integers(50, 9000)),
                'duration_mean': float(rng.integers(50, 900)),
                'peakExecutionMemory_max': int(rng.integers(1, 10**9)),
                'resultSize_max': int(rng.integers(1, 10**5)),
            }
            for metric in SUM_METRICS:
                # some totals of a job are zero, but not its task time
                is_zero = metric in zero_metrics or (metric != 'duration' and rng.random() < 0.15)
                job[f'{metric}_sum'] = 0 if is_zero else int(rng.integers(1, 10**9))
            if with_support:
                # each SQL has supported and unsupported rows
                job['Exec Is Supported'] = job_index % 2 == 0
            jobs.append(job)

    for sql_id, node_names in enumerate(sql_ops):
        for node_id, node_name in enumerate(node_names):
            ops.append({'appId': app_id, 'sqlID': sql_id, 'nodeID': node_id, 'nodeName': node_name,
                        'metricType': 'sum', 'max': 0, 'name': 'number of output rows', 'total': 1})
        if with_cache:
            for name in ['cache hits size', 'cache misses size']:
                # the sizes are over and under the bytes that the jobs of a SQL read
                ops.append({'appId': app_id, 'sqlID': sql_id, 'nodeID': 0, 'nodeName': node_names[0],
                            'metricType': 'size', 'max': 0, 'name': name, 'total': int(rng.integers(1, 10**9))})
        for _ in range(int(rng.integers(1, 3))):
            scan_time = 0 if 'scan_time' in zero_metrics else int(rng.integers(1, 10**4))
            data_sources.append({'appId': app_id, 'sqlID': sql_id, 'data_size': int(rng.integers(1, 10**9)),
                                 'scan_time': scan_time, 'decode_time': int(rng.integers(0, 10**4))})
        if sql_id % 2 == 0:
            failed_tasks.append({'appName': 'app', 'appId': app_id, 'sqlID': sql_id,
                                 'failed_tasks': int(rng.integers(1, 9))})
        sqls.append({'appId': app_id, 'sqlID': sql_id, 'Duration': int(rng.integers(100, 10**5)),
                     'description': f'query {sql_id}', **APP_ATTRIBUTES})

    failed_tasks_tbl = pd.DataFrame(failed_tasks, columns=['appName', 'appId', 'sqlID', 'failed_tasks'])
    return {
        'app_tbl': pd.DataFrame(sqls),
        'ops_tbl': pd.DataFrame(ops),
        'job_stage_agg_tbl': pd.DataFrame(jobs),
        'wholestage_tbl': pd.DataFrame(),
        'ds_tbl': pd.DataFrame(data_sources),
        'failed_tasks_tbl': failed_tasks_tbl.astype({'sqlID': int, 'failed_tasks': int}),
    }


def make_apps(seed: int = 0, **kwargs) -> Dict[str, Dict[str, pd.DataFrame]]:
    """The tables of three apps, by app: of one SQL, of the four SQLs of SQL_OPS, and of twelve SQLs."""
    rng = np.random.default_rng(seed)
    many_sql_ops = [SQL_OPS[i % len(SQL_OPS)][:1 + i % 3] for i in range(12)]
    return {
        'app-one': make_app(rng, 'app-one', SQL_OPS[:1], **kwargs),
        'app-four': make_app(rng, 'app-four', SQL_OPS, **kwargs),
        'app-twelve': make_app(rng, 'app-twelve', many_sql_ops, **kwargs),
    }


def sql_jobs(app_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The rows of task metrics that belong to the SQLs of an app's table."""
    jobs = app_tables['job_stage_agg_tbl']
    return jobs[jobs['sqlID'].isin(app_tables['app_tbl']['sqlID'])]


def ratio(numerator: float, denominator: float) -> float:
    """A ratio as the featurizer derives one: zero where it is undefined."""
    return numerator / denominator if denominator else 0.0


@pytest.fixture(name='default')
def _default():
    """The default featurizer, as the config loads it."""
    get_config(reload=True)
    return load_plugin(get_abs_path('default.py', 'featurizers'))


@pytest.fixture(name='featurize')
def _featurize(default, monkeypatch):
    """Extract the raw features of apps from their tables, in place of the profiler's CSV files."""
    def featurize(apps: Dict[str, Dict[str, pd.DataFrame]], level: Optional[str] = None,
                  qualtool_filter: Optional[str] = None) -> pd.DataFrame:
        def load_csv_files(toc, app_id, **_):  # pylint: disable=unused-argument
            return {name: table.copy() for name, table in apps[app_id].items()}

        monkeypatch.setattr(default, 'load_csv_files', load_csv_files)
        toc = pd.DataFrame({'appId': list(apps)})
        # the featurizer checks that there is support information, and reads it from the tables
        node_level_supp = pd.DataFrame() if qualtool_filter == 'stage' else None
        level_args = {'level': level} if level else {}
        features = default.extract_raw_features(toc, node_level_supp, qualtool_filter, **level_args)
        return features.reset_index(drop=True)

    return featurize


class TestDefaultFeaturizer(SparkRapidsToolsUT):
    """Test class for the default featurizer"""

    def test_sql_level(self, featurize, default):
        """Test that there is one row per SQL by default, with the features of the SQL"""
        apps = make_apps()
        sql_df = featurize(apps)
        pd.testing.assert_frame_equal(featurize(apps, level='sql'), sql_df)

        num_sqls = {'app-one': 1, 'app-four': 4, 'app-twelve': 12}
        assert sql_df.groupby('appId', sort=False).size().to_dict() == num_sqls
        for app_id, app_tables in apps.items():
            app_rows = sql_df[sql_df['appId'] == app_id]
            assert sorted(app_rows['sqlID']) == list(range(num_sqls[app_id]))
            assert sorted(app_rows['description']) == sorted(f'query {i}' for i in range(num_sqls[app_id]))
            jobs = sql_jobs(app_tables)
            for _, row in app_rows.iterrows():
                totals = jobs[jobs['sqlID'] == row['sqlID']].sum(numeric_only=True)
                assert row['numTasks_sum'] == totals['numTasks_sum']
                assert row['duration_sum'] == totals['duration_sum']
                assert row['duration_mean'] == pytest.approx(totals['duration_sum'] / totals['numTasks_sum'])
                assert row['input_bytesRead_mean'] == pytest.approx(
                    totals['input_bytesRead_sum'] / totals['numTasks_sum'])
                assert row['sw_bytesWrittenRatio'] == pytest.approx(
                    ratio(totals['sw_bytesWritten_sum'], totals['input_bytesRead_sum']))
                assert row['jvmGCTime_mean'] == pytest.approx(totals['jvmGCTime_sum'] / totals['numTasks_sum'])
        # an operator is a flag of the SQL, however many of its nodes have the operator
        ops_cols = [col for col in sql_df.columns if col.startswith('sqlOp_')]
        assert set(ops_cols) == set(SQL_OPS_COUNTS)
        assert set(np.unique(sql_df[ops_cols].to_numpy())) == {0, 1}
        assert (sql_df['cache_hit_ratio'] == 0.0).all()
        assert 'fraction_supported' not in sql_df
        # the sums that the features are derived from are not features
        assert not {f'{name}_sum' for name in default.BYTE_FEATURES + default.TIME_FEATURES} & set(sql_df.columns)

    def test_app_level_rows(self, featurize):
        """Test that there is one row per app at the app level, with the columns of the SQL-level rows"""
        apps = make_apps()
        sql_df = featurize(apps)
        app_df = featurize(apps, level='app')

        # one row per app, in the order of the SQL-level rows
        assert app_df['appId'].tolist() == ['app-four', 'app-one', 'app-twelve']
        assert app_df['appId'].tolist() == sql_df['appId'].unique().tolist()
        # an app-level row describes no single SQL
        assert app_df['sqlID'].tolist() == [-1, -1, -1]
        assert app_df['description'].tolist() == ['', '', '']
        assert app_df['appName'].tolist() == ['app', 'app', 'app']
        assert app_df['hasSqlID'].tolist() == [True, True, True]
        # the columns are those of the SQL-level rows, in their order and with their types
        assert app_df.columns.tolist() == sql_df.columns.tolist()
        assert app_df.dtypes.to_dict() == sql_df.dtypes.to_dict()
        assert not app_df.isna().any(axis=None)

        # the attributes of an app are those of each of its SQLs
        for name, value in APP_ATTRIBUTES.items():
            if name != 'startTime':
                assert (app_df[name] == value).all()

        # the app of one SQL has the features of its SQL
        one_sql = sql_df[sql_df['appId'] == 'app-one'].reset_index(drop=True)
        one_app = app_df[app_df['appId'] == 'app-one'].reset_index(drop=True)
        pd.testing.assert_frame_equal(one_app.drop(columns=['sqlID', 'description']),
                                      one_sql.drop(columns=['sqlID', 'description']))

    def test_app_level_totals(self, featurize):
        """Test that the totals of an app are the sums, minima and maxima over its SQL rows"""
        apps = make_apps()
        sql_rows = featurize(apps).groupby('appId', sort=False)
        app_df = featurize(apps, level='app').set_index('appId')

        reducers = {'sum': 0, 'min': 0, 'max': 0}
        for col in app_df.columns:
            reducer = col.rsplit('_', 1)[-1]
            if reducer in reducers:
                reducers[reducer] += 1
                expected = sql_rows[col].agg(reducer)
                pd.testing.assert_series_equal(app_df[col], expected, check_dtype=False)
        # the sums of the task and record counts, and the minima and maxima of the task metrics
        assert reducers == {'sum': 8, 'min': 2, 'max': 3}
        for col in ['Duration', 'failed_tasks', 'scan_time', 'decode_time', 'data_size']:
            pd.testing.assert_series_equal(app_df[col], sql_rows[col].sum(), check_dtype=False)

        # and so the totals of the app's jobs that belong to its SQLs
        for app_id, app_tables in apps.items():
            app, jobs = app_df.loc[app_id], sql_jobs(app_tables)
            for col in ['numTasks_sum', 'duration_sum', 'input_recordsRead_sum', 'sw_recordsWritten_sum']:
                assert app[col] == jobs[col].sum()
            assert app['duration_min'] == jobs['duration_min'].min()
            assert app['duration_max'] == jobs['duration_max'].max()
            assert app['jobStartTime_min'] == jobs['jobStartTime_min'].min()
            assert app['peakExecutionMemory_max'] == jobs['peakExecutionMemory_max'].max()
            assert app['resultSize_max'] == jobs['resultSize_max'].max()
            assert app['Duration'] == app_tables['app_tbl']['Duration'].sum()
            assert app['failed_tasks'] == app_tables['failed_tasks_tbl']['failed_tasks'].sum()
            for col in ['scan_time', 'decode_time', 'data_size']:
                assert app[col] == app_tables['ds_tbl'][col].sum()

    def test_app_level_operators(self, featurize):
        """Test that an operator of an app is the number of its SQLs that have the operator"""
        apps = make_apps()
        sql_df = featurize(apps)
        app_df = featurize(apps, level='app').set_index('appId')

        ops_cols = [col for col in app_df.columns if col.startswith('sqlOp_')]
        assert set(ops_cols) == set(SQL_OPS_COUNTS)
        assert app_df.loc['app-four', ops_cols].to_dict() == SQL_OPS_COUNTS
        assert app_df.loc['app-one', ops_cols].to_dict() == {
            **dict.fromkeys(SQL_OPS_COUNTS, 0), 'sqlOp_Scan parquet': 1, 'sqlOp_Filter': 1, 'sqlOp_Project': 1}
        pd.testing.assert_frame_equal(app_df[ops_cols], sql_df.groupby('appId', sort=False)[ops_cols].sum())
        for app_id, app_tables in apps.items():
            # the operator of a scan is named without its table
            nodes = app_tables['ops_tbl'].assign(
                op=app_tables['ops_tbl']['nodeName'].str.replace(r'^(Scan parquet) .*', r'\1', regex=True))
            expected = nodes.groupby('op')['sqlID'].nunique()
            for op, num_sqls in expected.items():
                assert app_df.loc[app_id, f'sqlOp_{op}'] == num_sqls

    def test_app_level_derived_features(self, featurize, default):
        """Test that the means, ratios and bandwidths of an app are derived from the totals of the app"""
        apps = make_apps()
        sql_df = featurize(apps)
        app_df = featurize(apps, level='app').set_index('appId')

        for app_id, app_tables in apps.items():
            app, totals = app_df.loc[app_id], sql_jobs(app_tables).sum(numeric_only=True)
            num_tasks, input_bytes = totals['numTasks_sum'], totals['input_bytesRead_sum']
            assert app['input_bytesRead_mean'] == pytest.approx(input_bytes / num_tasks)
            for name in default.BYTE_FEATURES:
                assert app[f'{name}_mean'] == pytest.approx(totals[f'{name}_sum'] / num_tasks)
                assert app[f'{name}Ratio'] == pytest.approx(ratio(totals[f'{name}_sum'], input_bytes))
            for name in default.TIME_FEATURES:
                assert app[f'{name}_mean'] == pytest.approx(totals[f'{name}_sum'] / num_tasks)
            assert app['duration_mean'] == pytest.approx(totals['duration_sum'] / num_tasks)
            assert app['duration_ratio'] == pytest.approx(
                totals['duration_sum'] / app_tables['app_tbl']['Duration'].sum())
            assert app['failed_tasks_ratio'] == pytest.approx(
                app_tables['failed_tasks_tbl']['failed_tasks'].sum() / num_tasks)
            assert app['scan_bw'] == pytest.approx(
                app_tables['ds_tbl']['data_size'].sum() / app_tables['ds_tbl']['scan_time'].sum())
            assert app['shuffle_read_bw'] == pytest.approx(
                ratio(totals['sr_totalBytesRead_sum'], totals['sr_fetchWaitTime_sum']))
            assert app['shuffle_write_bw'] == pytest.approx(
                ratio(totals['sw_bytesWritten_sum'], totals['sw_writeTime_sum']))
            assert app['cache_hit_ratio'] == 0.0

        # a feature that is derived from the totals of an app is not the mean of the feature over its SQLs
        many = app_df.drop(index='app-one')
        sql_means = sql_df.groupby('appId', sort=False).mean(numeric_only=True).drop(index='app-one')
        for col in ['input_bytesRead_mean', 'sw_bytesWrittenRatio', 'jvmGCTime_mean', 'duration_ratio', 'scan_bw']:
            assert not np.isclose(many[col], sql_means[col], rtol=1e-3).any()

    def test_app_level_zero_totals(self, featurize, default):
        """Test that the features of an app are zero where its totals leave them undefined"""
        apps = make_apps(zero_metrics=['input_bytesRead', 'sr_fetchWaitTime', 'sw_writeTime', 'scan_time'])
        app_df = featurize(apps, level='app')

        assert not app_df.isna().any(axis=None)
        zero_cols = [f'{name}Ratio' for name in default.BYTE_FEATURES]
        zero_cols += ['input_bytesRead_mean', 'sr_fetchWaitTime_mean', 'sw_writeTime_mean', 'shuffle_read_bw',
                      'shuffle_write_bw', 'scan_bw', 'scan_time']
        assert (app_df[zero_cols] == 0).all(axis=None)
        # the other totals of the same features are not zero
        many = app_df[app_df['appId'] == 'app-twelve']
        assert (many[['sr_totalBytesRead_mean', 'sw_bytesWritten_mean', 'data_size']] > 0).all(axis=None)

    def test_app_level_excludes_jobs_outside_sqls(self, featurize):
        """Test that the jobs that belong to no SQL, or to a SQL that is not in the app table, do not contribute to
        the features of the app"""
        apps = make_apps()
        only_sql_jobs = {
            app_id: {**app_tables, 'job_stage_agg_tbl': sql_jobs(app_tables)} for app_id, app_tables in apps.items()
        }
        for app_id, app_tables in apps.items():
            other_sql_ids = set(app_tables['job_stage_agg_tbl']['sqlID']) - set(app_tables['app_tbl']['sqlID'])
            assert other_sql_ids == {-1, len(app_tables['app_tbl'])}
            assert len(only_sql_jobs[app_id]['job_stage_agg_tbl']) < len(app_tables['job_stage_agg_tbl'])

        app_df = featurize(apps, level='app')
        expected = featurize(only_sql_jobs, level='app')

        pd.testing.assert_frame_equal(app_df, expected)

    def test_app_level_cache_hit_ratio(self, featurize):
        """Test that the cache hit ratio of an app is of the total cache sizes of its SQLs, and that the bytes read
        are corrected with the cache sizes of each SQL"""
        apps = make_apps(with_cache=True)
        sql_df = featurize(apps)
        app_df = featurize(apps, level='app')
        assert app_df.columns.tolist() == sql_df.columns.tolist()
        assert app_df.dtypes.to_dict() == sql_df.dtypes.to_dict()
        sql_df = sql_df.set_index(['appId', 'sqlID'])
        app_df = app_df.set_index('appId')

        num_corrected = 0
        for app_id, app_tables in apps.items():
            ops = app_tables['ops_tbl']
            hits = ops[ops['name'] == 'cache hits size'].groupby('sqlID')['total'].sum()
            misses = ops[ops['name'] == 'cache misses size'].groupby('sqlID')['total'].sum()
            app = app_df.loc[app_id]
            assert app['cache_hit_ratio'] == pytest.approx(hits.sum() / (hits.sum() + misses.sum()))
            # the ratio of each SQL
            for sql_id in hits.index:
                assert sql_df.loc[(app_id, sql_id), 'cache_hit_ratio'] == pytest.approx(
                    hits[sql_id] / (hits[sql_id] + misses[sql_id]))

            # each SQL read at least as many bytes as it looked up in the cache
            jobs = sql_jobs(app_tables)
            read = jobs.groupby('sqlID')['input_bytesRead_sum'].sum()
            corrected = np.maximum(read, hits + misses)
            num_corrected += (corrected > read).sum()
            assert app['input_bytesRead_mean'] == pytest.approx(corrected.sum() / jobs['numTasks_sum'].sum())
            assert app['sw_bytesWrittenRatio'] == pytest.approx(jobs['sw_bytesWritten_sum'].sum() / corrected.sum())
            assert app['input_bytesReadRatio'] == 1.0
        # some SQLs are corrected, and some are not
        assert 0 < num_corrected < len(sql_df)

    def test_app_level_fraction_supported(self, featurize):
        """Test that the supported fraction of an app is of the task time of the stages of its SQLs, when the
        featurizer is given the stage filter. load_profiles gives it the filter 'none' at the app level."""
        apps = make_apps(with_support=True)
        sql_df = featurize(apps, qualtool_filter='stage')
        app_df = featurize(apps, level='app', qualtool_filter='stage')
        assert app_df.columns.tolist() == sql_df.columns.tolist()
        assert app_df.dtypes.to_dict() == sql_df.dtypes.to_dict()
        sql_df = sql_df.set_index(['appId', 'sqlID'])
        app_df = app_df.set_index('appId')

        for app_id, app_tables in apps.items():
            stages = sql_jobs(app_tables)
            supported = stages[stages['Exec Is Supported']]
            assert 0 < len(supported) < len(stages)
            assert app_df.loc[app_id, 'fraction_supported'] == pytest.approx(
                supported['duration_sum'].sum() / stages['duration_sum'].sum())
            # the fraction of each SQL
            for sql_id, sql_stages in stages.groupby('sqlID'):
                sql_supported = sql_stages[sql_stages['Exec Is Supported']]
                assert sql_df.loc[(app_id, sql_id), 'fraction_supported'] == pytest.approx(
                    ratio(sql_supported['duration_sum'].sum(), sql_stages['duration_sum'].sum()))
        assert app_df['fraction_supported'].between(0.0, 1.0, inclusive='neither').all()

    def test_reduce_to_app(self, default):
        """Test that each total is reduced by the function in its name, and that a column without one is an error"""
        sql_totals = pd.DataFrame({
            'appId': ['app1', 'app1', 'app2', 'app1'],
            'appName': ['app', 'app', 'app', 'app'],
            'sqlID': [0, 1, 0, 2],
            'description': ['query 0', 'query 1', 'query 0', 'query 2'],
            'numTasks_sum': [10, 20, 5, 30],
            'duration_sum': [100, 500, 70, 300],
            'duration_min': [3.0, 1.0, 7.0, 2.0],
            'duration_max': [30.0, 90.0, 20.0, 50.0],
            'duration_mean': [10.0, 25.0, 14.0, 10.0],
            'sqlOp_Filter': [1, 0, 1, 1],
            'sqlOp_Scan parquet': [0, 0, 1, 0],
            'Duration': [1000, 2000, 400, 3000],
            'failed_tasks': [0.0, 2.0, 0.0, 1.0],
            'scan_time': [5, 6, 7, 8],
            'decode_time': [1, 0, 2, 3],
            'data_size': [100, 200, 300, 400],
            'cacheHitsSize_sum': [10.0, np.nan, 5.0, 30.0],
            'hasSqlID': [True, True, True, True],
            'appDuration': [9000, 9000, 800, 9000],
            'executorCores': [16, 16, 8, 16],
        })
        app_cols = ['appId', 'sqlID', 'Duration', 'description', 'appDuration', 'executorCores']

        app_totals = default._reduce_to_app(sql_totals, app_cols)  # pylint: disable=protected-access

        expected = pd.DataFrame({
            'appId': ['app1', 'app2'],
            'appName': ['app', 'app'],
            'numTasks_sum': [60, 5],
            'duration_sum': [900, 70],
            'duration_min': [1.0, 7.0],
            'duration_max': [90.0, 20.0],
            'sqlOp_Filter': [2, 1],
            'sqlOp_Scan parquet': [0, 1],
            'Duration': [6000, 400],
            'failed_tasks': [3.0, 0.0],
            'scan_time': [19, 7],
            'decode_time': [4, 2],
            'data_size': [700, 300],
            'cacheHitsSize_sum': [40.0, 5.0],
            'hasSqlID': [True, True],
            'appDuration': [9000, 800],
            'executorCores': [16, 8],
            'sqlID': [-1, -1],
            'description': ['', ''],
        })
        pd.testing.assert_frame_equal(app_totals, expected)

        # a column that is neither a total nor an attribute of the app
        for col in ['input_bytesRead_avg', 'executorRunTime_mean', 'stageIds', 'duration_ratio']:
            with pytest.raises(ValueError, match=f'No app-level reducer for the column {col}'):
                default._reduce_to_app(sql_totals.assign(**{col: 1.0}), app_cols)  # pylint: disable=protected-access

    def test_invalid_level(self, featurize):
        apps = make_apps()
        for level in ['stage', 'job', 'App']:
            with pytest.raises(ValueError, match="level must be 'sql' or 'app'"):
                featurize(apps, level=level)
