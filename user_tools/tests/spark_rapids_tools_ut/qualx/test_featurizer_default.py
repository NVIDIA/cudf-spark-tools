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

"""Test raw feature extraction in the default qualx featurizer using small synthetic profiler CSVs."""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from spark_rapids_tools.api_v1.report_loader import ReportLoader
from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.featurizers.default import (
    DS_KEY_COLS,
    DS_METRIC_COLS,
    expected_raw_features,
    extract_raw_features
)
from ..conftest import SparkRapidsToolsUT

DS_FEATURES = ['scan_bw', 'scan_time', 'decode_time', 'data_size']

DS_HEADER = [
    'sqlID', 'sql_plan_version', 'nodeId', 'format', 'buffer_time', 'scan_time', 'data_size', 'decode_time',
    'location', 'pushedFilters', 'schema', 'data_filters', 'partition_filters', 'from_final_plan'
]

TASK_METRICS = [
    'diskBytesSpilled_sum', 'duration_sum', 'duration_max', 'duration_min', 'duration_avg', 'executorCPUTime_sum',
    'executorDeserializeCPUTime_sum', 'executorDeserializeTime_sum', 'executorRunTime_sum', 'input_bytesRead_sum',
    'input_bytesRead_max', 'input_recordsRead_sum', 'jvmGCTime_sum', 'memoryBytesSpilled_sum',
    'output_bytesWritten_sum', 'output_recordsWritten_sum', 'peakExecutionMemory_max',
    'resultSerializationTime_sum', 'resultSize_max', 'sr_fetchWaitTime_sum', 'sr_localBlocksFetched_sum',
    'sr_localBytesRead_sum', 'sr_remoteBlocksFetched_sum', 'sr_remoteBytesRead_sum',
    'sr_remoteBytesReadToDisk_sum', 'sr_totalBytesRead_sum', 'sw_bytesWritten_sum', 'sw_recordsWritten_sum',
    'sw_writeTime_sum'
]

SQL_IDS = [0, 1]


def _task_metrics(num_tasks: int) -> Dict[str, int]:
    metrics = {col: 10 for col in TASK_METRICS}
    metrics.update({'duration_sum': 100 * num_tasks, 'input_bytesRead_sum': 1000, 'numTasks': num_tasks})
    return metrics


def _profiler_tables(app_id: str, ops: Dict[int, str]) -> Dict[str, pd.DataFrame]:
    """Minimal profiler tables for one app: one job and one stage per SQL, with one plan node per SQL."""
    app_info = pd.DataFrame([{
        'appName': 'app', 'appId': app_id, 'attemptId': 1, 'sparkUser': 'user', 'startTime': 1000,
        'endTime': 61000, 'exitCode': 0, 'duration': 60000, 'durationStr': '1.0 min', 'sparkRuntime': 'SPARK',
        'sparkVersion': '3.5.1', 'pluginEnabled': False, 'totalCoreSeconds': 240
    }])
    exec_info = pd.DataFrame([{
        'resourceProfileId': 0, 'numExecutors': 2, 'executorCores': 4, 'maxMem': 1024, 'maxOnHeapMem': 1024,
        'maxOffHeapMem': 0, 'executorMemory': 2048, 'numGpusPerExecutor': 0, 'executorOffHeap': 0,
        'taskCpu': 1.0, 'taskGpu': 0.0
    }])
    spark_props = pd.DataFrame([{'propertyName': 'spark.app.id', 'propertyValue': app_id}])
    sql_duration = pd.DataFrame([{
        'App ID': app_id, 'RootSqlID': sql_id, 'sqlID': sql_id, 'SQL Duration': 5000,
        'Contains Dataset or RDD Op': False, 'App Duration': 60000, 'Potential Problems': '',
        'Executor CPU Time Percent': 50.0
    } for sql_id in SQL_IDS])
    sql_metrics = pd.DataFrame([{
        'appID': app_id, 'sqlID': sql_id, 'description': f'query {sql_id}', 'Duration': 5000,
        'executorCPURatio': 50.0, **_task_metrics(4)
    } for sql_id in SQL_IDS])
    sql_to_stage = pd.DataFrame([{
        'sqlID': sql_id, 'jobID': sql_id, 'stageId': sql_id, 'stageAttemptId': 0, 'Stage Duration': 4000,
        'SQL Nodes(IDs)': f'{ops[sql_id]}(1)'
    } for sql_id in SQL_IDS])
    jobs = pd.DataFrame([{
        'jobID': sql_id, 'stageIds': str(sql_id), 'sqlID': sql_id, 'startTime': 2000, 'endTime': 6000
    } for sql_id in SQL_IDS])
    sql_ops = pd.DataFrame([{
        'sqlID': sql_id, 'nodeID': 1, 'nodeName': ops[sql_id], 'accumulatorId': 10 + sql_id,
        'name': 'number of output rows', 'min': 1, 'median': 1, 'max': 1, 'total': 4, 'metricType': 'sum',
        'stageIds': str(sql_id)
    } for sql_id in SQL_IDS])
    job_agg = pd.DataFrame([{'jobId': sql_id, 'Duration': 4000, **_task_metrics(4)} for sql_id in SQL_IDS])
    stage_agg = pd.DataFrame([{'stageId': sql_id, 'Duration': 4000, **_task_metrics(4)} for sql_id in SQL_IDS])
    for agg in [job_agg, stage_agg]:
        agg.insert(1, 'numTasks', agg.pop('numTasks'))
    return {
        'application_information': app_info,
        'executor_information': exec_info,
        'spark_properties': spark_props,
        'sql_duration_and_executor_cpu_time_percent': sql_duration,
        'sql_level_aggregated_task_metrics': sql_metrics,
        'sql_to_stage_information': sql_to_stage,
        'job_information': jobs,
        'sql_plan_metrics_for_application': sql_ops,
        'job_level_aggregated_task_metrics': job_agg,
        'stage_level_aggregated_task_metrics': stage_agg,
    }


def _scan_rows(rows: List[tuple]) -> pd.DataFrame:
    """Build data_source_information rows from (sqlID, nodeId, scan_time, data_size, decode_time) tuples."""
    return pd.DataFrame([{
        'sqlID': sql_id, 'sql_plan_version': 0, 'nodeId': node_id, 'format': 'Parquet', 'buffer_time': 0,
        'scan_time': scan_time, 'data_size': data_size, 'decode_time': decode_time,
        'location': 's3://bucket/table', 'pushedFilters': '', 'schema': 'a:int', 'data_filters': '[]',
        'partition_filters': '[]', 'from_final_plan': True
    } for sql_id, node_id, scan_time, data_size, decode_time in rows], columns=DS_HEADER)


def _write_app(root: str, app_id: str, ds_mode: str, scans: Optional[List[tuple]] = None,
               drop_tables: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Write one app's profiler CSVs and return its table-of-contents rows.

    ds_mode controls data_source_information.csv: 'absent', 'zero_byte', 'header_only' or 'rows'.
    """
    ops = {0: 'Scan parquet ' if scans else 'LocalTableScan', 1: 'LocalTableScan'}
    tables = _profiler_tables(app_id, ops)
    for name in drop_tables or []:
        tables.pop(name)
    app_dir = os.path.join(root, app_id)
    os.makedirs(app_dir)
    paths = {}
    for name, tbl in tables.items():
        paths[name] = os.path.join(app_dir, f'{name}.csv')
        tbl.to_csv(paths[name], index=False)
    ds_path = os.path.join(app_dir, 'data_source_information.csv')
    if ds_mode == 'zero_byte':
        open(ds_path, 'w', encoding='utf-8').close()  # pylint: disable=consider-using-with
    elif ds_mode == 'header_only':
        _scan_rows([]).to_csv(ds_path, index=False)
    elif ds_mode == 'rows':
        _scan_rows(scans).to_csv(ds_path, index=False)
    if ds_mode != 'absent':
        paths['data_source_information'] = ds_path
    return [{'filepath': path, 'ds_name': 'test', 'appId': app_id, 'table_name': name}
            for name, path in paths.items()]


def _extract(apps: List[dict], tmp_path, **kwargs) -> pd.DataFrame:
    toc = pd.DataFrame([row for app in apps for row in _write_app(str(tmp_path), **app)])
    return extract_raw_features(toc, node_level_supp=None, qualtool_filter=None, **kwargs)


def _assert_valid_features(features: pd.DataFrame, app_ids: List[str]) -> None:
    """Every (appId, sqlID) pair is kept and data source features are finite numbers."""
    expected_keys = sorted((app_id, sql_id) for app_id in app_ids for sql_id in SQL_IDS)
    assert sorted(zip(features['appId'], features['sqlID'])) == expected_keys
    for col in DS_FEATURES:
        assert pd.api.types.is_numeric_dtype(features[col]), col
        assert np.isfinite(features[col].to_numpy(dtype=float)).all(), col


def _ds_features(features: pd.DataFrame, app_id: str, sql_id: int) -> Dict[str, float]:
    row = features.loc[(features['appId'] == app_id) & (features['sqlID'] == sql_id)]
    assert len(row) == 1
    return {col: float(row[col].iloc[0]) for col in DS_FEATURES}


NO_SCAN = dict.fromkeys(DS_FEATURES, 0.0)
# sqlID 0 has two scan nodes: scan_time 10 + 30, data_size 100 + 300, decode_time 5 + 7
SCANS = [(0, 3, 10, 100, 5), (0, 4, 30, 300, 7)]
SCAN_SQL0 = {'scan_bw': 400 / 40, 'scan_time': 40.0, 'decode_time': 12.0, 'data_size': 400.0}


@pytest.mark.filterwarnings('error::FutureWarning')
class TestDefaultFeaturizer(SparkRapidsToolsUT):
    """Test extract_raw_features with and without data source information."""

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

    @pytest.mark.parametrize('ds_mode', ['absent', 'zero_byte', 'header_only'])
    def test_no_data_source_rows_single_app(self, tmp_path, ds_mode):
        features = _extract([{'app_id': 'app-1', 'ds_mode': ds_mode}], tmp_path)
        _assert_valid_features(features, ['app-1'])
        for sql_id in SQL_IDS:
            assert _ds_features(features, 'app-1', sql_id) == NO_SCAN

    def test_no_data_source_rows_multiple_apps(self, tmp_path):
        apps = [{'app_id': 'app-1', 'ds_mode': 'absent'},
                {'app_id': 'app-2', 'ds_mode': 'zero_byte'},
                {'app_id': 'app-3', 'ds_mode': 'header_only'}]
        features = _extract(apps, tmp_path)
        _assert_valid_features(features, ['app-1', 'app-2', 'app-3'])
        for app_id in ['app-1', 'app-2', 'app-3']:
            for sql_id in SQL_IDS:
                assert _ds_features(features, app_id, sql_id) == NO_SCAN

    @pytest.mark.parametrize('scan_first', [True, False])
    @pytest.mark.parametrize('ds_mode', ['absent', 'zero_byte', 'header_only'])
    def test_mixed_scan_and_no_scan_apps(self, tmp_path, ds_mode, scan_first):
        scan_app = {'app_id': 'app-scan', 'ds_mode': 'rows', 'scans': SCANS}
        no_scan_app = {'app_id': 'app-noscan', 'ds_mode': ds_mode}
        apps = [scan_app, no_scan_app] if scan_first else [no_scan_app, scan_app]
        features = _extract(apps, tmp_path)
        _assert_valid_features(features, ['app-scan', 'app-noscan'])
        assert _ds_features(features, 'app-scan', 0) == SCAN_SQL0
        assert _ds_features(features, 'app-scan', 1) == NO_SCAN
        for sql_id in SQL_IDS:
            assert _ds_features(features, 'app-noscan', sql_id) == NO_SCAN

    def test_scans_only(self, tmp_path):
        features = _extract([{'app_id': 'app-scan', 'ds_mode': 'rows', 'scans': SCANS}], tmp_path)
        _assert_valid_features(features, ['app-scan'])
        assert _ds_features(features, 'app-scan', 0) == SCAN_SQL0
        assert _ds_features(features, 'app-scan', 1) == NO_SCAN

    def test_malformed_data_source_rows_still_fail(self, tmp_path):
        # non-empty data source rows without the expected metric columns are not silently zero-filled
        app = {'app_id': 'app-scan', 'ds_mode': 'rows', 'scans': SCANS}
        toc = pd.DataFrame(_write_app(str(tmp_path), **app))
        ds_path = toc.loc[toc['table_name'] == 'data_source_information', 'filepath'].iloc[0]
        pd.read_csv(ds_path).drop(columns=['scan_time']).to_csv(ds_path, index=False)
        with pytest.raises(KeyError, match='scan_time'):
            extract_raw_features(toc, node_level_supp=None, qualtool_filter=None)

    @pytest.mark.parametrize('ds_mode', ['absent', 'rows'])
    def test_missing_required_table_falls_back(self, tmp_path, ds_mode):
        app = {'app_id': 'app-1', 'ds_mode': ds_mode, 'scans': SCANS if ds_mode == 'rows' else None,
               'drop_tables': ['sql_plan_metrics_for_application']}
        features = _extract([app], tmp_path)
        assert features.empty
        assert list(features.columns) == list(expected_raw_features)

    @pytest.mark.parametrize('ds_mode', ['absent', 'rows'])
    def test_all_unsupported_stages_falls_back(self, tmp_path, ds_mode):
        app = {'app_id': 'app-1', 'ds_mode': ds_mode, 'scans': SCANS if ds_mode == 'rows' else None}
        toc = pd.DataFrame(_write_app(str(tmp_path), **app))
        node_level_supp = pd.DataFrame({
            'App ID': ['app-1', 'app-1'], 'SQL ID': SQL_IDS, 'SQL Node Id': [1, 1], 'Exec Is Supported': [False, False]
        })
        features = extract_raw_features(toc, node_level_supp=node_level_supp, qualtool_filter='stage')
        assert features.empty
        assert list(features.columns) == list(expected_raw_features)
