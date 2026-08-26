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

"""Classify Spark runtime from accumulated Spark properties.

The scanner extracts Spark properties from event-log records and passes the
merged map to this module. This module only answers whether those properties
represent standard Spark or a cuDF plugin-enabled application.
"""

from typing import Mapping

from spark_rapids_tools.tools.eventlog_detector import markers as m
from spark_rapids_tools.tools.eventlog_detector.types import SparkRuntime


def _parse_bool(raw: str, default: bool) -> bool:
    """Parse Spark boolean strings with Scala-compatible fallback behavior.

    Scala's ``String.toBoolean`` accepts only ``true`` and ``false``. The
    Scala tools wrap that parse in ``Try(...).getOrElse(default)``, so values
    such as ``yes``, ``1``, or an empty string must return ``default`` rather
    than using Python truthiness.
    """
    stripped = raw.strip().lower()
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    return default


def _is_spark_rapids(props: Mapping[str, str]) -> bool:
    """Return true when Spark properties show the cuDF plugin is active."""
    plugins = props.get(m.GPU_PLUGIN_KEY, "")
    if m.GPU_PLUGIN_CLASS_SUBSTRING not in plugins:
        return False
    raw = props.get(m.GPU_ENABLED_KEY)
    if raw is None:
        return m.GPU_ENABLED_DEFAULT
    return _parse_bool(raw, default=m.GPU_ENABLED_DEFAULT)


def has_rapids_conf_markers(props: Mapping[str, str]) -> bool:
    """Return true when properties contain any cuDF plugin-related configuration.

    This is intentionally broader than ``_is_spark_rapids``. A disabled or
    incomplete cuDF plugin configuration is not classified as a cuDF plugin runtime, but its
    presence should prevent early CPU routing because later events may update
    the effective configuration.
    """
    if m.GPU_PLUGIN_CLASS_SUBSTRING in props.get(m.GPU_PLUGIN_KEY, ""):
        return True
    return m.GPU_ENABLED_KEY in props


def classify_runtime(props: Mapping[str, str]) -> SparkRuntime:
    """Classify accumulated Spark properties into the supported runtime enum."""
    if _is_spark_rapids(props):
        return SparkRuntime.SPARK_RAPIDS
    return SparkRuntime.SPARK
