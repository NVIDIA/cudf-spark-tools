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

"""Integration tests for ``eventlog_detector.detect_spark_runtime``."""
# pylint: disable=too-few-public-methods  # test classes naturally have few methods

import json
from pathlib import Path

import pytest

from spark_rapids_tools.storagelib import CspPath
from spark_rapids_tools.tools.eventlog_detector import detect_spark_runtime
from spark_rapids_tools.tools.eventlog_detector.types import (
    SparkRuntime,
    ToolExecution,
    UnsupportedInputError,
)


def env_update(props: dict) -> dict:
    return {
        "Event": "SparkListenerEnvironmentUpdate",
        "Spark Properties": props,
        "System Properties": {},
        "Classpath Entries": {},
        "JVM Information": {},
    }


def build_info() -> dict:
    return {
        "Event": "com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent",
        "sparkRapidsBuildInfo": {"version": "24.06.0"},
        "sparkRapidsJniBuildInfo": {},
        "cudfBuildInfo": {},
        "sparkRapidsPrivateBuildInfo": {},
    }


def sql_exec_start(modified_configs: dict) -> dict:
    return {
        "Event": "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart",
        "executionId": 0,
        "description": "",
        "details": "",
        "physicalPlanDescription": "",
        "sparkPlanInfo": {},
        "time": 0,
        "modifiedConfigs": modified_configs,
    }


def _write_plain_log(path: Path, events: list) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


class TestAcceptsStringPath:
    """Test that detect_spark_runtime accepts plain string paths."""

    def test_str_input_resolves(self, tmp_path):
        log = tmp_path / "eventlog"
        _write_plain_log(
            log,
            [
                {"Event": "SparkListenerLogStart", "Spark Version": "3.5.1"},
                {"Event": "SparkListenerApplicationStart", "App ID": "a", "App Name": "A"},
                env_update({"spark.master": "local"}),
            ],
        )
        result = detect_spark_runtime(str(log))
        assert result.tool_execution is ToolExecution.QUALIFICATION
        assert result.spark_runtime is SparkRuntime.SPARK


class TestRapidsLog:
    """Test detection on cuDF plugin event logs."""

    def test_build_info_event_classifies_as_profiling(self, tmp_path):
        log = tmp_path / "eventlog"
        _write_plain_log(
            log,
            [
                {"Event": "SparkListenerLogStart", "Spark Version": "3.5.1"},
                build_info(),
            ],
        )
        result = detect_spark_runtime(CspPath(str(log)))
        assert result.tool_execution is ToolExecution.PROFILING
        assert result.spark_runtime is SparkRuntime.SPARK_RAPIDS
        assert result.spark_version == "3.5.1"

    def test_env_update_with_plugin_classifies_as_profiling(self, tmp_path):
        log = tmp_path / "eventlog"
        _write_plain_log(
            log,
            [
                {"Event": "SparkListenerLogStart", "Spark Version": "3.5.1"},
                {"Event": "SparkListenerApplicationStart", "App ID": "g", "App Name": "G"},
                env_update({"spark.plugins": "com.nvidia.spark.SQLPlugin"}),
            ],
        )
        result = detect_spark_runtime(CspPath(str(log)))
        assert result.tool_execution is ToolExecution.PROFILING
        assert result.spark_runtime is SparkRuntime.SPARK_RAPIDS
        assert result.app_id == "g"


class TestCpuFastPath:
    """Test the default fast path for startup properties that look like plain Spark."""

    def test_env_update_without_rapids_markers_returns_qualification(self, tmp_path):
        log = tmp_path / "eventlog"
        _write_plain_log(
            log,
            [
                {"Event": "SparkListenerLogStart", "Spark Version": "3.5.1"},
                {"Event": "SparkListenerApplicationStart", "App ID": "c", "App Name": "C"},
                env_update({"spark.master": "local"}),
                sql_exec_start({"spark.plugins": "com.nvidia.spark.SQLPlugin"}),
            ],
        )
        result = detect_spark_runtime(str(log))
        assert result.tool_execution is ToolExecution.QUALIFICATION
        assert result.spark_runtime is SparkRuntime.SPARK
        assert "startup properties" in result.reason.lower()

    def test_fast_path_can_be_disabled(self, tmp_path):
        log = tmp_path / "eventlog"
        _write_plain_log(
            log,
            [
                env_update({"spark.master": "local"}),
                sql_exec_start({"spark.plugins": "com.nvidia.spark.SQLPlugin"}),
            ],
        )
        result = detect_spark_runtime(str(log), allow_cpu_fast_path=False)
        assert result.tool_execution is ToolExecution.PROFILING
        assert result.spark_runtime is SparkRuntime.SPARK_RAPIDS


class TestCapHit:
    """Test detection when the event budget is exhausted before env-update."""

    def test_no_env_update_before_cap_is_unknown(self, tmp_path):
        log = tmp_path / "eventlog"
        _write_plain_log(
            log,
            [{"Event": "SparkListenerLogStart", "Spark Version": "3.5.1"}] * 10,
        )
        result = detect_spark_runtime(str(log), max_events_scanned=5)
        assert result.tool_execution is ToolExecution.UNKNOWN
        assert result.spark_runtime is None


class TestOssRolling:
    """Test detection on Apache Spark rolling event-log directories."""

    def test_rapids_signal_in_later_rolled_file(self, tmp_path):
        d = tmp_path / "eventlog_v2_app-1"
        d.mkdir()
        _write_plain_log(
            d / "events_1_app-1",
            [
                {"Event": "SparkListenerLogStart", "Spark Version": "3.5.1"},
                {"Event": "SparkListenerApplicationStart", "App ID": "d", "App Name": "D"},
                env_update({"spark.rapids.sql.enabled": "false"}),
            ],
        )
        _write_plain_log(d / "events_2_app-1", [build_info()])
        result = detect_spark_runtime(CspPath(str(d)))
        assert result.tool_execution is ToolExecution.PROFILING
        assert result.spark_runtime is SparkRuntime.SPARK_RAPIDS
        assert result.event_log_path.endswith("/events_2_app-1")

    def test_cpu_fast_path_applies_to_rolling_dir(self, tmp_path):
        d = tmp_path / "eventlog_v2_app-1"
        d.mkdir()
        _write_plain_log(d / "events_1_app-1", [env_update({"spark.master": "local"})])
        _write_plain_log(d / "events_2_app-1", [sql_exec_start({"spark.plugins": "com.nvidia.spark.SQLPlugin"})])
        result = detect_spark_runtime(CspPath(str(d)))
        assert result.tool_execution is ToolExecution.QUALIFICATION
        assert result.spark_runtime is SparkRuntime.SPARK
        assert result.event_log_path.endswith("/events_1_app-1")
        assert "startup properties" in result.reason.lower()


class TestUnsupportedInput:
    """Test that unsupported input shapes raise the expected error."""

    def test_non_oss_rolling_dir_raises(self, tmp_path):
        d = tmp_path / "non_oss_rolling"
        d.mkdir()
        (d / "eventlog").write_bytes(b"")
        (d / "eventlog-2021-06-14--18-00.gz").write_bytes(b"")
        with pytest.raises(UnsupportedInputError):
            detect_spark_runtime(CspPath(str(d)))


class TestSourcePathPreserved:
    """Test that source_path echoes the original input string."""

    def test_source_path_equals_input_string(self, tmp_path):
        log = tmp_path / "eventlog"
        _write_plain_log(log, [env_update({"spark.master": "local"})])
        input_str = str(log)
        result = detect_spark_runtime(input_str)
        assert result.source_path == input_str
