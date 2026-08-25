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

"""Unit tests for ``eventlog_detector.scanner``."""

import json
from pathlib import Path
from typing import List

from spark_rapids_tools.storagelib import CspPath
from spark_rapids_tools.tools.eventlog_detector.scanner import (
    scan_events,
    scan_events_across,
)
from spark_rapids_tools.tools.eventlog_detector.types import Termination


def env_update(props: dict) -> str:
    return json.dumps(
        {
            "Event": "SparkListenerEnvironmentUpdate",
            "Spark Properties": props,
            "System Properties": {},
            "Classpath Entries": {},
            "JVM Information": {},
        }
    )


def log_start(version: str = "3.5.1") -> str:
    return json.dumps({"Event": "SparkListenerLogStart", "Spark Version": version})


def app_start(app_id: str = "app-1", app_name: str = "App") -> str:
    return json.dumps(
        {
            "Event": "SparkListenerApplicationStart",
            "App ID": app_id,
            "App Name": app_name,
        }
    )


def rapids_build_info() -> str:
    return json.dumps(
        {
            "Event": "com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent",
            "sparkRapidsBuildInfo": {"version": "24.06.0"},
            "sparkRapidsJniBuildInfo": {},
            "cudfBuildInfo": {},
            "sparkRapidsPrivateBuildInfo": {},
        }
    )


def sql_exec_start(modified_configs: dict) -> str:
    return json.dumps(
        {
            "Event": "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart",
            "executionId": 0,
            "description": "x",
            "details": "",
            "physicalPlanDescription": "",
            "sparkPlanInfo": {},
            "time": 0,
            "modifiedConfigs": modified_configs,
        }
    )


class TestScanEvents:
    """Tests for scan_events scanning a single event stream."""

    def test_build_info_event_is_decisive_before_env_update(self):
        lines = iter([log_start(), rapids_build_info(), app_start()])
        result = scan_events(lines, budget=100)
        assert result.rapids_build_info_seen is True
        assert result.termination is Termination.DECISIVE
        assert result.events_scanned == 2

    def test_env_update_with_gpu_is_decisive(self):
        # Scala tools and the cuDF plugin both default spark.rapids.sql.enabled to true,
        # so the plugin marker alone is enough to classify it as a cuDF plugin runtime.
        lines = iter(
            [
                log_start(),
                app_start(),
                env_update({"spark.plugins": "com.nvidia.spark.SQLPlugin"}),
            ]
        )
        result = scan_events(lines, budget=100)
        assert result.env_update_seen is True
        assert result.app_id == "app-1"
        assert result.spark_version == "3.5.1"
        assert result.termination is Termination.DECISIVE

    def test_cpu_fast_path_stops_at_env_update_by_default(self):
        lines = iter(
            [
                log_start(),
                app_start(),
                env_update({"spark.master": "local"}),
                sql_exec_start({"spark.plugins": "com.nvidia.spark.SQLPlugin"}),
            ]
        )
        result = scan_events(lines, budget=100)
        assert result.env_update_seen is True
        assert result.termination is Termination.CPU_FAST_PATH
        assert result.events_scanned == 3

    def test_cpu_fast_path_can_be_disabled(self):
        lines = iter(
            [
                log_start(),
                app_start(),
                env_update({"spark.master": "local"}),
                sql_exec_start({"spark.plugins": "com.nvidia.spark.SQLPlugin"}),
            ]
        )
        result = scan_events(lines, budget=100, allow_cpu_fast_path=False)
        assert result.env_update_seen is True
        assert result.termination is Termination.DECISIVE
        assert result.events_scanned == 4

    def test_fast_path_ignored_when_rapids_marker_present(self):
        lines = iter(
            [
                env_update({
                    "spark.plugins": "com.nvidia.spark.SQLPlugin",
                    "spark.rapids.sql.enabled": "false",
                }),
                sql_exec_start({"spark.rapids.sql.enabled": "true"}),
            ]
        )
        result = scan_events(lines, budget=100)
        assert result.termination is Termination.DECISIVE
        assert result.spark_properties["spark.rapids.sql.enabled"] == "true"

    def test_no_env_update_within_budget_is_cap_hit(self):
        # Budget less than the number of events, none of them env-update.
        lines = iter([log_start()] * 5)
        result = scan_events(lines, budget=2)
        assert result.env_update_seen is False
        assert result.termination is Termination.CAP_HIT

    def test_no_env_update_to_eof_is_exhausted_without_env(self):
        lines = iter([log_start(), app_start()])
        result = scan_events(lines, budget=100)
        assert result.env_update_seen is False
        assert result.termination is Termination.EXHAUSTED

    def test_malformed_json_lines_are_skipped(self):
        lines = iter(
            [
                "not-json-at-all",
                log_start(),
                "",
                app_start(),
                env_update({"spark.master": "local"}),
            ]
        )
        result = scan_events(lines, budget=100)
        assert result.env_update_seen is True
        assert result.app_id == "app-1"

    def test_sql_start_classifies_after_full_modified_config_merge(self):
        lines = iter(
            [
                env_update({"spark.rapids.sql.enabled": "false"}),
                sql_exec_start({
                    "spark.plugins": "com.nvidia.spark.SQLPlugin",
                    "spark.rapids.sql.enabled": "false",
                }),
            ]
        )
        result = scan_events(lines, budget=100)
        assert result.termination is Termination.EXHAUSTED
        assert result.spark_properties["spark.plugins"] == "com.nvidia.spark.SQLPlugin"
        assert result.spark_properties["spark.rapids.sql.enabled"] == "false"


def _write(path: Path, lines: List[str]) -> CspPath:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CspPath(str(path))


class TestScanEventsAcross:
    """Tests for scan_events_across scanning across multiple files."""

    def test_gpu_signal_in_second_file_is_decisive(self, tmp_path):
        f1 = _write(
            tmp_path / "events_1_app-1",
            [log_start(), app_start(), env_update({"spark.rapids.sql.enabled": "false"})],
        )
        f2 = _write(tmp_path / "events_2_app-1", [rapids_build_info()])
        result = scan_events_across([f1, f2], budget=100)
        assert result.termination is Termination.DECISIVE
        assert result.last_scanned_path == str(f2)

    def test_cpu_fast_path_applies_across_files_when_no_rapids_markers(self, tmp_path):
        f1 = _write(tmp_path / "events_1_app-1", [env_update({"spark.master": "local"})])
        f2 = _write(tmp_path / "events_2_app-1", [sql_exec_start({"spark.plugins": "com.nvidia.spark.SQLPlugin"})])
        result = scan_events_across([f1, f2], budget=100)
        assert result.termination is Termination.CPU_FAST_PATH
        assert result.last_scanned_path == str(f1)

    def test_cpu_fast_path_skips_when_rapids_marker_present_across_files(self, tmp_path):
        f1 = _write(
            tmp_path / "events_1_app-1",
            [env_update({"spark.rapids.sql.enabled": "false"})],
        )
        f2 = _write(
            tmp_path / "events_2_app-1",
            [sql_exec_start({
                "spark.plugins": "com.nvidia.spark.SQLPlugin",
                "spark.rapids.sql.enabled": "true",
            })],
        )
        result = scan_events_across([f1, f2], budget=100)
        assert result.termination is Termination.DECISIVE
        assert result.last_scanned_path == str(f2)

    def test_shared_budget_applied_across_files(self, tmp_path):
        # 3 events in first file, 3 in second. Budget = 4. Second file stops
        # after one event, before any GPU signal.
        f1 = _write(tmp_path / "events_1_app-1", [log_start(), app_start(), env_update({
            "spark.rapids.sql.enabled": "false",
        })])
        f2 = _write(
            tmp_path / "events_2_app-1",
            [
                sql_exec_start({"spark.master": "still-cpu"}),
                sql_exec_start({"spark.plugins": "com.nvidia.spark.SQLPlugin"}),
                sql_exec_start({"x": "y"}),
            ],
        )
        result = scan_events_across([f1, f2], budget=4)
        assert result.termination is Termination.CAP_HIT

    def test_all_files_exhausted_returns_exhausted(self, tmp_path):
        f1 = _write(tmp_path / "events_1_app-1", [env_update({"spark.master": "local"})])
        result = scan_events_across([f1], budget=100, allow_cpu_fast_path=False)
        assert result.termination is Termination.EXHAUSTED
