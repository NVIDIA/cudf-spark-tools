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
#
# Usage:
#   jq --arg fixture_id db173-photon-merge-fixture -c \
#     -f scripts/sanitize-databricks-eventlog.jq EVENTLOG \
#     | zstd -19 -T0 -o SANITIZED_EVENTLOG.zstd
#
# This filter is shared by the DBR 17.3 Photon qualification fixture and the independent RAPIDS
# GPU profiling fixture. It retains every event and redacts identifying fields in place.
# Review every generated fixture for sensitive fields because future event types may require
# additional sanitization rules.
#
# Regenerate the DBR 17.3 qualification goldens from core/ with:
#   mvn test \
#     -Dsuites=com.nvidia.spark.rapids.tool.qualification.QualificationNoSparkSuite \
#     -Dtools.qual.test.generate.golden.enable=true
# Review the generated photon_db_17_3 files under golden-sets/357/qual before copying them to
# src/test/resources/QualificationExpectations/photon_db_17_3.
#
# The synthetic Databricks cluster tags are required because Databricks runtime detection uses
# clusterAllTags, clusterId, and clusterName in addition to runtime-specific properties.

def sanitize_string:
  gsub("/Workspace/[^ \\t\\r\\n\"]+"; "/Workspace/REDACTED")
  | gsub("abfss://[^ \\t\\r\\n\",)]+"; "abfss://REDACTED")
  | gsub("dbfs:/[^ \\t\\r\\n\",)]+"; "dbfs:/REDACTED")
  | gsub("s3://[^ \\t\\r\\n\\\",)\\]]+"; "s3://dummy-s3-bucket/REDACTED")
  | gsub("https?://[^ \\t\\r\\n\",)]+"; "https://REDACTED")
  | gsub("([0-9]{1,3}\\.){3}[0-9]{1,3}"; "192.0.2.1")
  | gsub("[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";
      "00000000-0000-0000-0000-000000000000");

def sanitize_strings:
  walk(if type == "string" then sanitize_string else . end);

def keep_spark_property:
  . == "spark.app.id"
  or . == "spark.app.name"
  or . == "spark.app.startTime"
  or . == "spark.master"
  or . == "spark.plugins"
  or . == "spark.sql.extensions"
  or . == "spark.executor.memory"
  or . == "spark.executor.cores"
  or . == "spark.executor.instances"
  or startswith("spark.dynamicAllocation.")
  or . == "spark.databricks.clusterUsageTags.sparkVersion"
  or . == "spark.databricks.clusterUsageTags.effectiveSparkVersion"
  or . == "spark.databricks.clusterUsageTags.sparkImageLabel"
  or . == "spark.databricks.clusterUsageTags.runtimeEngine"
  or . == "spark.databricks.clusterUsageTags.cloudProvider"
  or . == "spark.databricks.clusterUsageTags.clusterNodeType"
  or . == "spark.databricks.clusterUsageTags.driverNodeType"
  or . == "spark.databricks.clusterUsageTags.clusterWorkers"
  or . == "spark.databricks.clusterUsageTags.clusterTargetWorkers";

if .Event == "SparkListenerEnvironmentUpdate" then
  .["Spark Properties"] |= with_entries(select(.key | keep_spark_property))
  | .["Spark Properties"]["spark.databricks.clusterUsageTags.clusterAllTags"] =
      ([{"key":"Vendor","value":"Databricks"},
        {"key":"ClusterName","value":$fixture_id},
        {"key":"ClusterId","value":$fixture_id}] | tojson)
  | .["Spark Properties"]["spark.databricks.clusterUsageTags.clusterId"] =
      $fixture_id
  | .["Spark Properties"]["spark.databricks.clusterUsageTags.clusterName"] =
      $fixture_id
  | .["Hadoop Properties"] = {}
  | .["System Properties"] = {}
  | .["Classpath Entries"] = {}
elif .Event == "SparkListenerApplicationStart" then
  .User = "test-user"
elif .Event == "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart" then
  .modifiedConfigs = {}
  | .jobTags = []
  | .jobGroupId = $fixture_id
  | .queryId = "00000000-0000-0000-0000-000000000000"
  | .details = ""
elif .Event == "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd" then
  .queryId = "00000000-0000-0000-0000-000000000000"
elif .Event == "org.apache.spark.sql.connect.service.SparkListenerConnectServiceStarted" then
  .hostAddress = "192.0.2.1"
elif .Event == "SparkListenerJobStart" then
  .Properties |= with_entries(select(.key == "spark.sql.execution.id"))
elif .Event == "SparkListenerStageSubmitted" then
  .Properties |= with_entries(select(.key == "spark.sql.execution.id"))
elif .Event == "SparkListenerExecutorAdded" then
  .["Executor Info"].Host = "executor.example.invalid"
  | .["Executor Info"]["Log Urls"] = {}
  | .["Executor Info"].Attributes = {}
elif (.Event == "SparkListenerBlockManagerAdded"
    or .Event == "SparkListenerBlockManagerRemoved") then
  .["Block Manager ID"].Host = "executor.example.invalid"
elif .Event == "com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent" then
  .sparkRapidsBuildInfo.user = "test-user"
  | .sparkRapidsJniBuildInfo.user = "test-user"
  | .cudfBuildInfo.user = "test-user"
  | .sparkRapidsPrivateBuildInfo.user = "test-user"
elif .Event == "SparkListenerTaskEnd" then
  .["Task Info"].Host = "executor.example.invalid"
  # Spark replays TaskEnd accumulator updates from Update. Value repeats the final value, while
  # these metadata fields are not consumed by the qualification fixture. Removing them keeps the
  # checked-in event log below the repository's per-file size limit without changing its reports.
  | .["Task Info"].Accumulables |=
      map(del(.Metadata, .["Count Failed Values"], .Internal, .Value))
  # Exception messages embed storage paths and request details that the URI rules cannot match.
  | if .["Task End Reason"].Reason == "ExceptionFailure" then
      .["Task End Reason"].Description = "REDACTED"
      | .["Task End Reason"]["Full Stack Trace"] =
          "\(.["Task End Reason"]["Class Name"]): REDACTED"
    else . end
elif (.Event == "SparkListenerTaskStart" or .Event == "SparkListenerTaskGettingResult") then
  .["Task Info"].Host = "executor.example.invalid"
else
  .
end
| if (
    .Event == "SparkListenerEnvironmentUpdate"
    or .Event == "SparkListenerApplicationStart"
    or .Event == "SparkListenerJobStart"
    or .Event == "SparkListenerStageSubmitted"
    or .Event == "SparkListenerStageCompleted"
    or .Event == "SparkListenerExecutorAdded"
    or .Event == "SparkListenerBlockManagerAdded"
    or .Event == "com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent"
    or .Event == "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart"
    or .Event == "org.apache.spark.sql.execution.ui.SparkListenerSQLAdaptiveExecutionUpdate"
  ) then sanitize_strings else . end
