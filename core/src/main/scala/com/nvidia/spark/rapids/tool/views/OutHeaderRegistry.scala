/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.tool.views

/**
 * This class is used to register the output headers for each view.
 * This centralized registration is a memory optimization to avoid creating a copy of the header
 * for each object instantiation.
 * The utility object is a centralized way to compared to creating a separate companion object for
 * each case class.
 * Note that the key lookup depends on literal strings for performance reasons.
 * The idea that we do not want to use reflections such as classForName
 * (or more broadly, classOf[YourClass].getName). the latter would impose a significant performance
 * hit considering the number of rows to be processed for each file.
 */
object OutHeaderRegistry {
  val outputHeaders: Map[String, Array[String]] = Map(
    "ExecutorInfoProfileResult" ->
      Array("resourceProfileId", "numExecutors", "executorCores", "maxMem", "maxOnHeapMem",
        "maxOffHeapMem", "executorMemory", "numGpusPerExecutor", "executorOffHeap", "taskCpu",
        "taskGpu"),
    "JobInfoProfileResult" ->
      Array("jobID", "stageIds", "sqlID", "startTime", "endTime"),
    "SQLCleanAndAlignIdsProfileResult" ->
      Array("sqlID"),
    "SQLPlanInfoProfileResult" ->
      Array("sqlID", "SparkPlanInfoTruncated"),
    "SQLPlanGraph" ->
      // This is the table that shows the SQL plan graph. It dumps basic information
      // about the nodes in each SQL. See "SQLPlanGraphProfileResult" for more details.
      Array("sqlID", "sqlPlanVersion", "nodeID", "nodeName", "description", "sinkNodes",
        "stageIds"),
    "SQLStageInfoProfileResult" ->
      Array("sqlID", "jobID", "stageId", "stageAttemptId", "Stage Duration", "SQL Nodes(IDs)"),
    "RapidsJarProfileResult" ->
      Array("Rapids4Spark jars"),
    "DataSourceProfileResult" ->
      Array("sqlID", "sql_plan_version", "nodeId", "format", "buffer_time", "scan_time",
        "data_size", "decode_time", "location", "pushedFilters", "schema", "data_filters",
        "partition_filters", "from_final_plan"),
    "DriverLogUnsupportedOperators" ->
      Array("operatorName", "count", "reason"),
    "AppStatusResult" ->
      Array("Event Log", "Status", "App ID", "Attempt ID", "App Name", "Description"),
    "SQLAccumProfileResults" ->
      Array("sqlID", "nodeID", "nodeName", "accumulatorId", "name", "min", "median", "max",
        "total", "metricType", "stageIds"),
    "AccumProfileResults" ->
      Array("stageId", "accumulatorId", "name", "min", "median", "max", "total"),
    "BlockManagerRemovedProfileResult" ->
      Array("executorId", "time"),
    "ExecutorsRemovedProfileResult" ->
      Array("executorId", "time", "reason"),
    "UnsupportedOpsProfileResult" ->
      Array("sqlID", "nodeID", "nodeName", "nodeDescription", "reason"),
    "AppInfoProfileResults" ->
      Array("appName", "appId", "attemptId", "sparkUser", "startTime", "endTime", "exitCode",
        "duration", "durationStr", "sparkRuntime", "sparkVersion", "pluginEnabled",
        "totalCoreSeconds"),
    "AppLevelRecommendationSignalsProfileResult" ->
      Array("appId", "numScanStagesWithGpuOom", "numGpuShuffleStagesWithContainerOom"),
    "AppLogPathProfileResults" ->
      Array("appName", "appId", "eventLogPath"),
    "FailedTaskProfileResults" ->
      Array(
          // Begin metadata of the failed task.
          "stageId", "stageAttemptId", "taskId", "attempt", "failureReason",
          // some useful columns that could be used to categorize failures
          "taskStatus",
          "taskType",
          "speculative",
          // Begin metrics the failed task if any.
          "duration",
          "diskBytesSpilled",
          "executorCPUTimeNS",
          "executorDeserializeCPUTimeNS",
          "executorDeserializeTime",
          "executorRunTime",
          "input_bytesRead",
          "input_recordsRead",
          "jvmGCTime",
          "memoryBytesSpilled",
          "output_bytesWritten",
          "output_recordsWritten",
          "peakExecutionMemory",
          "resultSerializationTime",
          "resultSize",
          "sr_fetchWaitTime",
          "sr_localBlocksFetched",
          "sr_localBytesRead",
          "sr_remoteBlocksFetched",
          "sr_remoteBytesRead",
          "sr_remoteBytesReadToDisk",
          "sr_totalBytesRead",
          "sw_bytesWritten",
          "sw_recordsWritten",
          "sw_writeTimeNS"),
    "FailedStagesProfileResults" ->
      Array("stageId", "attemptId", "name", "numTasks", "failureReason"),
    "FailedJobsProfileResults" ->
      Array("jobID", "sqlID", "jobResult", "failureReason"),
    "JobAggTaskMetricsProfileResult" ->
      Array("jobId",
        "numTasks",
        "Duration",
        "diskBytesSpilled_sum",
        "duration_sum",
        "duration_max",
        "duration_min",
        "duration_avg",
        "executorCPUTime_sum",
        "executorDeserializeCPUTime_sum",
        "executorDeserializeTime_sum",
        "executorRunTime_sum",
        "input_bytesRead_sum",
        "input_bytesRead_max",
        "input_recordsRead_sum",
        "jvmGCTime_sum",
        "memoryBytesSpilled_sum",
        "output_bytesWritten_sum",
        "output_recordsWritten_sum",
        "peakExecutionMemory_max",
        "resultSerializationTime_sum",
        "resultSize_max",
        "sr_fetchWaitTime_sum",
        "sr_localBlocksFetched_sum",
        "sr_localBytesRead_sum",
        "sr_remoteBlocksFetched_sum",
        "sr_remoteBytesRead_sum",
        "sr_remoteBytesReadToDisk_sum",
        "sr_totalBytesRead_sum",
        "sw_bytesWritten_sum",
        "sw_recordsWritten_sum",
        "sw_writeTime_sum"),
    "StageAggTaskMetricsProfileResult" ->
      Array("stageId",
        "numTasks",
        "Duration",
        "diskBytesSpilled_sum",
        "duration_sum",
        "duration_max",
        "duration_min",
        "duration_avg",
        "executorCPUTime_sum",
        "executorDeserializeCPUTime_sum",
        "executorDeserializeTime_sum",
        "executorRunTime_sum",
        "input_bytesRead_sum",
        "input_bytesRead_max",
        "input_recordsRead_sum",
        "jvmGCTime_sum",
        "memoryBytesSpilled_sum",
        "output_bytesWritten_sum",
        "output_recordsWritten_sum",
        "peakExecutionMemory_max",
        "resultSerializationTime_sum",
        "resultSize_max",
        "sr_fetchWaitTime_sum",
        "sr_localBlocksFetched_sum",
        "sr_localBytesRead_sum",
        "sr_remoteBlocksFetched_sum",
        "sr_remoteBytesRead_sum",
        "sr_remoteBytesReadToDisk_sum",
        "sr_totalBytesRead_sum",
        "sw_bytesWritten_sum",
        "sw_recordsWritten_sum",
        "sw_writeTime_sum"),
    "StageDiagnosticResult" ->
      Array("appName",
        "appId",
        "stageId",
        "stageDurationMs",
        "numTasks",
        "memoryBytesSpilledMBMin",
        "memoryBytesSpilledMBMedian",
        "memoryBytesSpilledMBMax",
        "memoryBytesSpilledMBTotal",
        "diskBytesSpilledMBMin",
        "diskBytesSpilledMBMedian",
        "diskBytesSpilledMBMax",
        "diskBytesSpilledMBTotal",
        "inputBytesReadMin",
        "inputBytesReadMedian",
        "inputBytesReadMax",
        "inputBytesReadTotal",
        "outputBytesWrittenMin",
        "outputBytesWrittenMedian",
        "outputBytesWrittenMax",
        "outputBytesWrittenTotal",
        "shuffleReadBytesMin",
        "shuffleReadBytesMedian",
        "shuffleReadBytesMax",
        "shuffleReadBytesTotal",
        "shuffleWriteBytesMin",
        "shuffleWriteBytesMedian",
        "shuffleWriteBytesMax",
        "shuffleWriteBytesTotal",
        "shuffleReadFetchWaitTimeMin",
        "shuffleReadFetchWaitTimeMedian",
        "shuffleReadFetchWaitTimeMax",
        "shuffleReadFetchWaitTimeTotal",
        "shuffleWriteWriteTimeMin",
        "shuffleWriteWriteTimeMedian",
        "shuffleWriteWriteTimeMax",
        "shuffleWriteWriteTimeTotal",
        "gpuSemaphoreWaitTimeTotal",
        "SQL Nodes(IDs)"),
    "SQLTaskAggMetricsProfileResult" ->
      Array("appID",
        "sqlID",
        "description",
        "numTasks",
        "Duration",
        "executorCPURatio",
        "diskBytesSpilled_sum",
        "duration_sum",
        "duration_max",
        "duration_min",
        "duration_avg",
        "executorCPUTime_sum",
        "executorDeserializeCPUTime_sum",
        "executorDeserializeTime_sum",
        "executorRunTime_sum",
        "input_bytesRead_sum",
        "input_bytesRead_max",
        "input_recordsRead_sum",
        "jvmGCTime_sum",
        "memoryBytesSpilled_sum",
        "output_bytesWritten_sum",
        "output_recordsWritten_sum",
        "peakExecutionMemory_max",
        "resultSerializationTime_sum",
        "resultSize_max",
        "sr_fetchWaitTime_sum",
        "sr_localBlocksFetched_sum",
        "sr_localBytesRead_sum",
        "sr_remoteBlocksFetched_sum",
        "sr_remoteBytesRead_sum",
        "sr_remoteBytesReadToDisk_sum",
        "sr_totalBytesRead_sum",
        "sw_bytesWritten_sum",
        "sw_recordsWritten_sum",
        "sw_writeTime_sum"),
    "IODiagnosticResult" ->
      Array("appName",
        "appId",
        "sqlId",
        "stageId",
        "stageDurationMs",
        "nodeId",
        "nodeName",
        "outputRowsMin",
        "outputRowsMedian",
        "outputRowsMax",
        "outputRowsTotal",
        "scanTimeMin",
        "scanTimeMedian",
        "scanTimeMax",
        "scanTimeTotal",
        "outputBatchesMin",
        "outputBatchesMedian",
        "outputBatchesMax",
        "outputBatchesTotal",
        "bufferTimeMin",
        "bufferTimeMedian",
        "bufferTimeMax",
        "bufferTimeTotal",
        "shuffleWriteTimeMin",
        "shuffleWriteTimeMedian",
        "shuffleWriteTimeMax",
        "shuffleWriteTimeTotal",
        "fetchWaitTimeMin",
        "fetchWaitTimeMedian",
        "fetchWaitTimeMax",
        "fetchWaitTimeTotal",
        "gpuDecodeTimeMin",
        "gpuDecodeTimeMedian",
        "gpuDecodeTimeMax",
        "gpuDecodeTimeTotal"),
    "IOAnalysisProfileResult" ->
      Array("appID", "sqlID", "input_bytesRead_sum", "input_recordsRead_sum",
        "output_bytesWritten_sum", "output_recordsWritten_sum", "diskBytesSpilled_sum",
        "memoryBytesSpilled_sum", "sr_totalBytesRead_sum", "sw_bytesWritten_sum"),
    "SQLDurationExecutorTimeProfileResult" ->
      Array("App ID",
        "RootSqlID",
        "sqlID",
        "SQL Duration",
        "Contains Dataset or RDD Op",
        "App Duration",
        "Potential Problems",
        "Executor CPU Time Percent"),
    "ShuffleSkewProfileResult" ->
      Array("stageId", "stageAttemptId", "taskId", "attempt",
        "taskDurationSec", "avgDurationSec", "taskShuffleReadMB", "avgShuffleReadMB",
        "taskPeakMemoryMB", "successful", "reason"),
    "RapidsPropertyProfileResult" ->
      Array("propertyName", "propertyValue"),
    "WholeStageCodeGenResults" ->
      Array("sqlID", "nodeID", "SQL Node", "Child Node", "Child NodeID"),
    "WriteOpProfileResult" ->
      Array("sqlID", "sqlPlanVersion", "nodeId", "fromFinalPlan", "execName", "format",
          "location", "tableName", "dataBase", "outputColumns", "writeMode",
          "partitionColumns", "compressionOption", "fullDescription"),
    "ConnectSessionProfileResult" ->
      Array("appID", "sessionId", "userId", "startTime", "endTime", "durationMs",
        "operationCount"),
    "ConnectOperationProfileResult" ->
      Array("appID", "operationId", "sessionId", "userId", "jobTag",
        "startTime", "finishTime", "closeTime", "failTime", "cancelTime", "durationMs",
        "status", "errorMessage",
        "sqlIds", "jobIds", "statementFile", "statementTruncated"),
    "StageAggGpuMetricsProfileResult" ->
      Array("stageId", "numTasks", "metricName", "unit", "sum", "max", "avg"),
    "SQLAggGpuMetricsProfileResult" ->
      Array("sqlId", "metricName", "unit", "sum", "max", "avg"),
    "AppAggGpuMetricsProfileResult" ->
      Array("appId", "metricName", "unit", "sum", "max", "avg")
  ) // End of outputHeaders map initialization
}
