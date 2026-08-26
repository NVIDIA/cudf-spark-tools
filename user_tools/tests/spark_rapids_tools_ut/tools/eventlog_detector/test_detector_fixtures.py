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

"""Anchor tests against fixtures already shipped in the Scala core.

These are not a full parity sweep. They catch regressions on a small
curated set covering each decisive execution decision.
"""

from pathlib import Path

import pytest

from spark_rapids_tools.tools.eventlog_detector import detect_spark_runtime
from spark_rapids_tools.tools.eventlog_detector.types import SparkRuntime, ToolExecution


REPO_ROOT = Path(__file__).resolve().parents[5]
CORE_FIXTURES = REPO_ROOT / "core" / "src" / "test" / "resources"


@pytest.mark.parametrize(
    "relative_path,expected_execution,expected_runtime",
    [
        (
            "spark-events-profiling/eventlog-gpu-dsv2.zstd",
            ToolExecution.PROFILING,
            SparkRuntime.SPARK_RAPIDS,
        ),
        (
            # The Profiling tool can process CPU logs; this fixture lives under
            # profiling resources but has no cuDF plugin runtime markers.
            "spark-events-profiling/eventlog_dsv2.zstd",
            ToolExecution.QUALIFICATION,
            SparkRuntime.SPARK,
        ),
        (
            "spark-events-qualification/eventlog_same_app_id_1.zstd",
            ToolExecution.QUALIFICATION,
            SparkRuntime.SPARK,
        ),
    ],
)
def test_detector_matches_expected_execution_on_scala_fixture(
    relative_path: str, expected_execution: ToolExecution, expected_runtime: SparkRuntime
) -> None:
    fixture = CORE_FIXTURES / relative_path
    if not fixture.exists():
        pytest.skip(f"fixture not available: {fixture}")
    # Fixtures are ~small; a generous budget keeps this test decisive.
    result = detect_spark_runtime(str(fixture), max_events_scanned=5000)
    assert result.tool_execution is expected_execution, result.reason
    assert result.spark_runtime is expected_runtime, result.reason
