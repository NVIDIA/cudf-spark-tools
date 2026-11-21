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

"""Test data source features in the default qualx featurizer using small synthetic profiler CSVs."""
import os

import numpy as np
import pandas as pd
import pytest

from spark_rapids_tools.api_v1.report_loader import ReportLoader
from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.featurizers.default import DS_KEY_COLS, DS_METRIC_COLS, extract_raw_features
from ..conftest import SparkRapidsToolsUT

DS_FEATURES = ['scan_bw'] + DS_METRIC_COLS
SQL_IDS = [0, 1]
TASK_METRICS = [
    'diskBytesSpilled_sum', 'duration_sum', 'executorCPUTime_sum', 'executorDeserializeCPUTime_sum',
    'executorDeserializeTime_sum', 'executorRunTime_sum', 'input_bytesRead_sum', 'jvmGCTime_sum',
    'memoryBytesSpilled_sum', 'output_bytesWritten_sum', 'sr_fetchWaitTime_sum', 'sr_localBytesRead_sum',
    'sr_remoteBytesRead_sum', 'sr_remoteBytesReadToDisk_sum', 'sr_totalBytesRead_sum', 'sw_bytesWritten_sum',
    'sw_writeTime_sum'
]
# sqlID 0 has two scan nodes: (sqlID, nodeId, scan_time, data_size, decode_time)
SCANS = [(0, 3, 10, 100, 5), (0, 4, 30, 300, 7)]
SCAN_SQL0 = {'scan_bw': 400 / 40, 'scan_time': 40.0, 'decode_time': 12.0, 'data_size': 400.0}
NO_SCAN = dict.fromkeys(DS_FEATURES, 0.0)


def _rows(**cols) -> pd.DataFrame:
    """One row per sqlID; callables receive the sqlID."""
    return pd.DataFrame([{k: v(sql_id) if callable(v) else v for k, v in cols.items()} for sql_id in SQL_IDS])


def _write_app(root: str, app_id: str, ds_mode: str) -> list:
    """Write minimal profiler CSVs for one app (one job, stage and plan node per SQL); return its toc rows.

    ds_mode controls data_source_information.csv: 'absent', 'zero_byte', 'header_only', 'scans', 'malformed'
    (missing scan_time), 'unparseable' (a row with extra fields) or 'truncated' (a row missing decode_time).
    """
    metrics = {col: 10 for col in TASK_METRICS}
    tables = {
        'application_information': pd.DataFrame([{
            'appName': 'app', 'appId': app_id, 'startTime': 1000, 'sparkRuntime': 'SPARK',
            'sparkVersion': '3.5.1', 'pluginEnabled': False}]),
        'executor_information': pd.DataFrame([{
            'resourceProfileId': 0, 'numExecutors': 2, 'executorCores': 4, 'maxMem': 1024, 'maxOnHeapMem': 1024,
            'maxOffHeapMem': 0, 'executorMemory': 2048, 'numGpusPerExecutor': 0, 'executorOffHeap': 0,
            'taskCpu': 1.0, 'taskGpu': 0.0}]),
        'sql_duration_and_executor_cpu_time_percent': _rows(**{
            'App ID': app_id, 'sqlID': lambda s: s, 'App Duration': 60000, 'Potential Problems': '',
            'Contains Dataset or RDD Op': False}),
        'sql_level_aggregated_task_metrics': _rows(
            appID=app_id, sqlID=lambda s: s, description=lambda s: f'query {s}', Duration=5000, **metrics),
        'sql_to_stage_information': _rows(**{'sqlID': lambda s: s, 'jobID': lambda s: s, 'stageId': lambda s: s,
                                             'SQL Nodes(IDs)': 'Project(1)'}),
        'job_information': _rows(jobID=lambda s: s, sqlID=lambda s: s, startTime=2000),
        'sql_plan_metrics_for_application': _rows(
            sqlID=lambda s: s, nodeID=1, nodeName='Project', name='number of output rows', max=1, total=4,
            metricType='sum', stageIds=str),
        'job_level_aggregated_task_metrics': _rows(jobId=lambda s: s, numTasks=4, Duration=4000, **metrics),
        'stage_level_aggregated_task_metrics': _rows(stageId=lambda s: s, numTasks=4, Duration=4000, **metrics),
    }
    if ds_mode in ('scans', 'malformed'):
        tables['data_source_information'] = pd.DataFrame(
            SCANS, columns=['sqlID', 'nodeId', 'scan_time', 'data_size', 'decode_time']).assign(format='Parquet')
        if ds_mode == 'malformed':
            tables['data_source_information'] = tables['data_source_information'].drop(columns=['scan_time'])
    elif ds_mode == 'header_only':
        tables['data_source_information'] = pd.DataFrame(columns=['sqlID', 'scan_time', 'data_size', 'decode_time'])

    app_dir = os.path.join(root, app_id)
    os.makedirs(app_dir)
    paths = {name: os.path.join(app_dir, f'{name}.csv') for name in tables}
    for name, tbl in tables.items():
        tbl.to_csv(paths[name], index=False)
    header = 'sqlID,scan_time,data_size,decode_time\n0,10,100,5\n'
    raw_ds = {'zero_byte': '', 'unparseable': header + '1,10,100,5,EXTRA,FIELD\n', 'truncated': header + '1,10,100\n'}
    if ds_mode in raw_ds:
        paths['data_source_information'] = os.path.join(app_dir, 'data_source_information.csv')
        with open(paths['data_source_information'], 'w', encoding='utf-8') as f:
            f.write(raw_ds[ds_mode])
    return [{'filepath': p, 'ds_name': 'test', 'appId': app_id, 'table_name': n} for n, p in paths.items()]


def _extract(tmp_path, apps: dict, qualtool_filter: str = None) -> pd.DataFrame:
    toc = pd.DataFrame([row for app_id, mode in apps.items() for row in _write_app(str(tmp_path), app_id, mode)])
    node_level_supp = None
    if qualtool_filter == 'stage':
        # the qualification path: every plan node (nodeID 1) is supported
        node_level_supp = pd.DataFrame([{'App ID': app_id, 'SQL ID': sql_id, 'SQL Node Id': 1,
                                         'Exec Is Supported': True} for app_id in apps for sql_id in SQL_IDS])
    return extract_raw_features(toc, node_level_supp=node_level_supp, qualtool_filter=qualtool_filter)


def _ds_features(features: pd.DataFrame) -> dict:
    """Data source features keyed by (appId, sqlID), after checking they are unique per key and finite numbers."""
    assert not features.duplicated(DS_KEY_COLS).any()
    for col in DS_FEATURES:
        assert pd.api.types.is_numeric_dtype(features[col]), col
        assert np.isfinite(features[col].to_numpy(dtype=float)).all(), col
    return {(row.appId, row.sqlID): {col: float(getattr(row, col)) for col in DS_FEATURES}
            for row in features.itertuples()}


@pytest.mark.filterwarnings('error::FutureWarning')
class TestDefaultFeaturizer(SparkRapidsToolsUT):
    """Test data source features produced by extract_raw_features."""

    @pytest.fixture(autouse=True)
    def reload_config(self):
        get_config(reload=True)

    def test_data_source_columns_match_core_report(self):
        # guards against an upstream rename/removal of a column the featurizer aggregates
        # (appId is added by the featurizer, it is not a data_source_information column)
        table = ReportLoader().report_defs['coreRawMetrics'].get_table_defs()['coreRawDataSourceInformationCSV']
        report_columns = {col.name for col in table.columns}
        required = (set(DS_KEY_COLS) - {'appId'}) | set(DS_METRIC_COLS)
        assert required <= report_columns, f'missing from core report: {required - report_columns}'

    @pytest.mark.parametrize('qualtool_filter', [None, 'stage'])
    def test_apps_without_data_source_rows(self, tmp_path, qualtool_filter):
        # no app in the batch has scans: previously failed with KeyError: 'appId'
        apps = {'app-1': 'absent', 'app-2': 'zero_byte', 'app-3': 'header_only'}
        features = _ds_features(_extract(tmp_path, apps, qualtool_filter))
        assert features == {(app_id, sql_id): NO_SCAN for app_id in apps for sql_id in SQL_IDS}

    @pytest.mark.parametrize('scan_first', [True, False])
    def test_mixed_scan_and_no_scan_apps(self, tmp_path, scan_first):
        apps = {'app-scan': 'scans', 'app-noscan': 'header_only'}
        if not scan_first:
            apps = dict(reversed(apps.items()))
        features = _ds_features(_extract(tmp_path, apps))
        assert features == {
            ('app-scan', 0): SCAN_SQL0, ('app-scan', 1): NO_SCAN,
            ('app-noscan', 0): NO_SCAN, ('app-noscan', 1): NO_SCAN,
        }

    @pytest.mark.parametrize('bad_mode', ['malformed', 'unparseable', 'truncated'])
    def test_bad_data_source_file_falls_back(self, tmp_path, caplog, bad_mode):
        # a bad data_source_information.csv makes its app fall back to 1.0 instead of getting zero scan features
        features = _ds_features(_extract(tmp_path / 'mixed', {'app-bad': bad_mode, 'app-noscan': 'absent'}))
        assert features == {('app-noscan', sql_id): NO_SCAN for sql_id in SQL_IDS}
        assert _extract(tmp_path / 'alone', {'app-bad': bad_mode}).empty
        fallbacks = [r.getMessage() for r in caplog.records if 'Predicted speedup will be 1.0' in r.getMessage()]
        assert len(fallbacks) == 2
        assert all('for app-bad. Reason:' in msg and 'data_source_information' in msg for msg in fallbacks)
        if bad_mode == 'unparseable':
            assert all('Expected 4 fields in line 3, saw 6' in msg for msg in fallbacks)
