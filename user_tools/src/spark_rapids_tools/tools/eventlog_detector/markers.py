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

"""Property keys and constants used by the runtime detector.

Each block carries a Scala source reference so the two implementations
can be kept in sync when the Scala detection rules change.
"""

# GPU (SPARK_RAPIDS) markers.
# Scala: org/apache/spark/sql/rapids/tool/ToolUtils.scala :: isPluginEnabled
GPU_PLUGIN_KEY: str = "spark.plugins"
GPU_PLUGIN_CLASS_SUBSTRING: str = "com.nvidia.spark.SQLPlugin"
GPU_ENABLED_KEY: str = "spark.rapids.sql.enabled"
# Defaults to true when missing or unparseable.
GPU_ENABLED_DEFAULT: bool = True

# cuDF plugin 24.06+ marker.
# Scala: com/nvidia/spark/rapids/SparkRapidsBuildInfoEvent.scala
EVENT_SPARK_RAPIDS_BUILD_INFO: str = "com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent"
EVENT_SPARK_RAPIDS_BUILD_INFO_SHORTNAME: str = "SparkRapidsBuildInfoEvent"

# Apache Spark rolling event-log directory layout.
# Scala: com/nvidia/spark/rapids/tool/EventLogPathProcessor.scala :: isEventLogDir
OSS_EVENT_LOG_DIR_PREFIX: str = "eventlog_v2_"
OSS_EVENT_LOG_FILE_PREFIX: str = "events_"

# Spark listener event names consumed by the scanner.
EVENT_LOG_START: str = "SparkListenerLogStart"
EVENT_APPLICATION_START: str = "SparkListenerApplicationStart"
EVENT_ENVIRONMENT_UPDATE: str = "SparkListenerEnvironmentUpdate"
EVENT_SQL_EXECUTION_START: str = "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart"
# Unqualified event name accepted by the scanner for compatibility.
EVENT_SQL_EXECUTION_START_SHORTNAME: str = "SparkListenerSQLExecutionStart"
