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

"""Top-level event log runtime detector."""

from typing import Optional, Union

from spark_rapids_tools.storagelib import CspPath
from spark_rapids_tools.tools.eventlog_detector.classifier import classify_runtime
from spark_rapids_tools.tools.eventlog_detector.resolver import resolve_event_log_files
from spark_rapids_tools.tools.eventlog_detector.scanner import scan_events_across
from spark_rapids_tools.tools.eventlog_detector.types import (
    DetectionResult,
    SparkRuntime,
    Termination,
    ToolExecution,
)


def detect_spark_runtime(
    event_log: Union[str, CspPath],
    *,
    max_events_scanned: int = 500,
    allow_cpu_fast_path: bool = True,
) -> DetectionResult:
    """Classify a single-app event log into a tool execution decision.

    Returns ``PROFILING`` when a cuDF plugin marker is found, ``QUALIFICATION`` when
    the log appears to be OSS Spark/CPU, and ``UNKNOWN`` when the bounded scan
    cannot make a decision.

    ``max_events_scanned`` caps CPU/IO cost. Logs that do not expose a cuDF plugin
    marker or ``SparkListenerEnvironmentUpdate`` within the cap remain
    ``UNKNOWN``.

    ``allow_cpu_fast_path`` enables early CPU routing when startup properties
    contain no cuDF plugin markers. Disable it to require EOF before returning
    ``QUALIFICATION``.
    """
    # Keep the caller's input verbatim in source_path (cloud URI schemes
    # would otherwise be stripped by CspPath normalisation).
    source_path = event_log if isinstance(event_log, str) else str(event_log)
    path = event_log if isinstance(event_log, CspPath) else CspPath(str(event_log))
    _, files = resolve_event_log_files(path)

    scan = scan_events_across(
        files,
        budget=max_events_scanned,
        allow_cpu_fast_path=allow_cpu_fast_path,
    )

    runtime: Optional[SparkRuntime]
    if scan.rapids_build_info_seen:
        runtime = SparkRuntime.SPARK_RAPIDS
    elif scan.env_update_seen:
        runtime = classify_runtime(scan.spark_properties)
    else:
        runtime = None

    if runtime is SparkRuntime.SPARK_RAPIDS:
        tool_execution = ToolExecution.PROFILING
        reason = f"decisive: classified as {runtime.value}"
    elif scan.termination is Termination.CPU_FAST_PATH and runtime is SparkRuntime.SPARK:
        tool_execution = ToolExecution.QUALIFICATION
        reason = "startup properties classify as SPARK with no cuDF plugin markers"
    elif scan.termination is Termination.EXHAUSTED and scan.env_update_seen:
        tool_execution = ToolExecution.QUALIFICATION
        reason = "walked full log, no cuDF plugin signal"
    else:
        tool_execution = ToolExecution.UNKNOWN
        reason = (
            "no decisive signal within bounded scan"
            if scan.env_update_seen
            else "no SparkListenerEnvironmentUpdate reached"
        )

    resolved_path = scan.last_scanned_path or (str(files[0]) if files else source_path)
    return DetectionResult(
        tool_execution=tool_execution,
        spark_runtime=runtime,
        app_id=scan.app_id,
        spark_version=scan.spark_version,
        event_log_path=resolved_path,
        source_path=source_path,
        reason=reason,
    )
