/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.profiling

import scala.collection.Map

import com.nvidia.spark.rapids.tool.analysis.StatisticsMetrics
import com.nvidia.spark.rapids.tool.views.OutHeaderRegistry

import org.apache.spark.resource.{ExecutorResourceRequest, TaskResourceRequest}
import org.apache.spark.sql.rapids.tool.store.{AccumMetaRef, SparkPlanInfoTruncated, TaskModel}
import org.apache.spark.sql.rapids.tool.util.{SparkRuntime, StringUtils}

/**
 * This is a warehouse to store all Classes
 * used for profiling and qualification.
 */
trait ProfileResult {
  def outputHeaders: Array[String] = Array.empty[String]
  def convertToSeq(): Array[String]
  def convertToCSVSeq(): Array[String]
}

case class ExecutorInfoProfileResult(resourceProfileId: Int,
    numExecutors: Int, executorCores: Int, maxMem: Long, maxOnHeapMem: Long,
    maxOffHeapMem: Long, executorMemory: Option[Long], numGpusPerExecutor: Option[Long],
    executorOffHeap: Option[Long], taskCpu: Option[Double],
    taskGpu: Option[Double]) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("ExecutorInfoProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(resourceProfileId.toString, numExecutors.toString,
      executorCores.toString, maxMem.toString, maxOnHeapMem.toString,
      maxOffHeapMem.toString, executorMemory.map(_.toString).orNull,
      numGpusPerExecutor.map(_.toString).orNull,
      executorOffHeap.map(_.toString).orNull, taskCpu.map(_.toString).orNull,
      taskGpu.map(_.toString).orNull)
  }
  override def convertToCSVSeq(): Array[String] = convertToSeq()
}

class JobInfoClass(val jobID: Int,
    val stageIds: Seq[Int],
    val sqlID: Option[Long],
    val properties: scala.collection.Map[String, String],
    val startTime: Long,
    var endTime: Option[Long],
    var jobResult: Option[String],
    var failedReason: Option[String],
    var duration: Option[Long],
    var gpuMode: Boolean)

case class JobInfoProfileResult(
    jobID: Int,
    stageIds: Seq[Int],
    sqlID: Option[Long],
    startTime: Long,
    endTime: Option[Long]) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("JobInfoProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    val stageIdStr = s"[${stageIds.mkString(",")}]"
    Array(jobID.toString,
      stageIdStr,
      sqlID.map(_.toString).orNull,
      startTime.toString,
      endTime.map(_.toString).orNull)
  }

  override def convertToCSVSeq(): Array[String] = {
    val stageIdStr = s"[${stageIds.mkString(",")}]"
    Array(jobID.toString,
      StringUtils.reformatCSVString(stageIdStr),
      sqlID.map(_.toString).orNull,
      startTime.toString,
      endTime.map(_.toString).orNull)
  }
}

case class SQLCleanAndAlignIdsProfileResult(
    sqlID: Long) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLCleanAndAlignIdsProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString)
  }
  override def convertToCSVSeq(): Array[String] = convertToSeq()
}

/**
 * This result class is used in the Qual/Prof Views to
 * display a SQL ID to truncated Spark Plan Info mapping
 * extracted from the SparkListenerSQLExecutionStart event.
 * @param sqlID The SQL ID associated with the profile result.
 * @param sparkPlanInfo The truncated Spark plan information.
 */
case class SQLPlanInfoProfileResult(sqlID: Long, sparkPlanInfo: SparkPlanInfoTruncated)
  extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLPlanInfoProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString, sparkPlanInfo.toString)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(sqlID.toString, StringUtils.reformatCSVString(sparkPlanInfo.toString))
  }
}

/**
 * This result class is used in the SQL graph report that displays
 * the full list of operators executed within a plan.
 * @param sqlID the SQL ID
 * @param planVersion the plan version (commonly the final plan)
 * @param nodeID the node ID
 * @param nodeName the node name
 * @param nodeDesc the node description. The caller should cleanup the description and escape the
 *                 special characters.
 * @param sinkNodes A comma separated list of adjacent node Ids. Those nodes are the destination of
 *                  the edge going out of the current node.
 * @param stageIds The stage IDs that the node belongs to.
 *                 This typically a raw representation using the accumulable IDs.
 */
case class SQLPlanGraphProfileResult(
    sqlID: Long,
    planVersion: Int,
    nodeID: Long,
    nodeName: String,
    nodeDesc: String, // node desc should be processed to escape characters
    sinkNodes: Iterable[Long],
    stageIds: Iterable[Int]) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLPlanGraph")
  }
  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString, planVersion.toString, nodeID.toString,
      nodeName, nodeDesc, sinkNodes.mkString(","), stageIds.mkString(","))
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(sqlID.toString, planVersion.toString, nodeID.toString,
      StringUtils.reformatCSVString(nodeName),
      StringUtils.reformatCSVString(nodeDesc),
      StringUtils.quoteCSVString(sinkNodes.mkString(",")),
      StringUtils.quoteCSVString(stageIds.mkString(",")))
  }
}

case class SQLStageInfoProfileResult(
    sqlID: Long,
    jobID: Int,
    stageId: Int,
    stageAttemptId: Int,
    duration: Option[Long],
    nodeNames: Seq[String]) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLStageInfoProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString, jobID.toString, stageId.toString,
      stageAttemptId.toString, duration.map(_.toString).orNull,
      nodeNames.mkString(","))
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(sqlID.toString, jobID.toString, stageId.toString,
      stageAttemptId.toString, duration.map(_.toString).orNull,
      StringUtils.reformatCSVString(nodeNames.mkString(",")))
  }
}

case class RapidsJarProfileResult(jar: String)  extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("RapidsJarProfileResult")
  }
  override def convertToSeq(): Array[String] = {
    Array(jar)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(jar))
  }
}

case class DataSourceProfileResult(sqlID: Long, version: Int, nodeId: Long,
    format: String, buffer_time: Long, scan_time: Long, data_size: Long,
    decode_time: Long, location: String, pushedFilters: String, schema: String,
    dataFilters: String, partitionFilters: String, fromFinalPlan: Boolean) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("DataSourceProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString, version.toString, nodeId.toString, format,
      buffer_time.toString, scan_time.toString, data_size.toString, decode_time.toString,
      location, pushedFilters, schema, dataFilters, partitionFilters, fromFinalPlan.toString)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(sqlID.toString, version.toString, nodeId.toString,
      StringUtils.reformatCSVString(format), buffer_time.toString, scan_time.toString,
      data_size.toString, decode_time.toString, StringUtils.reformatCSVString(location),
      StringUtils.reformatCSVString(pushedFilters), StringUtils.reformatCSVString(schema),
      StringUtils.reformatCSVString(dataFilters), StringUtils.reformatCSVString(partitionFilters),
      fromFinalPlan.toString)
  }
}

case class DriverLogUnsupportedOperators(
    operatorName: String, count: Int, reason: String) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("DriverLogUnsupportedOperators")
  }

  override def convertToSeq(): Array[String] = {
    Array(operatorName, count.toString, reason)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(operatorName), count.toString,
      StringUtils.reformatCSVString(reason))
  }
}

// Case class representing status summary information for a particular application.
case class AppStatusResult(
    path: String,
    status: String,
    message: String,
    appId: String,
    attemptId: String,
    appName: String) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("AppStatusResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(path, status, appId, attemptId, appName, message)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(path, StringUtils.quoteCSVString(status), StringUtils.reformatCSVString(appId),
      attemptId, StringUtils.reformatCSVString(appName),
      StringUtils.reformatCSVString(message))
  }
}

object AppStatusResult {
  def build(path: String,
      status: String,
      message: Option[String] = None,
      appId: Option[String] = None,
      attemptId: Option[Int] = None,
      appName: Option[String] = None) : AppStatusResult = {
    val processedMsg =
      // process the message to escape special characters if any
      message.map { m =>
        // Escape newlines and other special characters for CSV output
        StringUtils.renderStr(m, doEscapeMetaCharacters = true, maxLength = 0)
      }.getOrElse("N/A")
    val appIdStr = appId match {
      case Some(id) => id
      case None => "N/A"
    }
    val attemptIdStr = attemptId match {
      case Some(id) => id.toString
      case None => "N/A"
    }
    val appNameStr = appName match {
      case Some(name) => name
      case None => "N/A"
    }
    AppStatusResult(path, status, processedMsg, appIdStr, attemptIdStr, appNameStr)
  }
}

// note that some things might not be set until after sqlMetricsAggregation called
class SQLExecutionInfoClass(
    val sqlID: Long,
    val rootExecutionID: Option[Long],
    val description: String,
    val details: String,
    val startTime: Long,
    var endTime: Option[Long],
    var duration: Option[Long],
    var hasDatasetOrRDD: Boolean,
    var sqlCpuTimePercent: Double = -1,
    // Per-SQL-execution config overrides from spark.conf.set() calls.
    // Captured from SparkListenerSQLExecutionStart.modifiedConfigs (Spark 3.3+).
    // Empty for Spark 3.2.x event logs or when no runtime config changes were made.
    // This stores only the overrides, not the full config set.
    val modifiedConfigs: Map[String, String] = Map.empty) {
  def setDsOrRdd(value: Boolean): Unit = {
    hasDatasetOrRDD = value
  }
}

case class SQLAccumProfileResults(
    sqlID: Long,
    nodeID: Long,
    nodeName: String,
    accumulatorId: Long,
    name: String,
    min: Long,
    median: Long,
    max: Long,
    total: Long,
    metricType: String,
    stageIds: Set[Int]) extends ProfileResult {

  private val stageIdsStr = stageIds.mkString(",")

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLAccumProfileResults")
  }

  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString,
      nodeID.toString,
      nodeName,
      accumulatorId.toString,
      name,
      min.toString,
      median.toString,
      max.toString,
      total.toString,
      metricType,
      stageIdsStr)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(sqlID.toString,
      nodeID.toString,
      StringUtils.reformatCSVString(nodeName),
      accumulatorId.toString,
      StringUtils.reformatCSVString(name),
      min.toString,
      median.toString,
      max.toString,
      total.toString,
      StringUtils.reformatCSVString(metricType),
      StringUtils.reformatCSVString(stageIdsStr))
  }
}

case class AccumProfileResults(
    stageId: Int,
    accMetaRef: AccumMetaRef,
    min: Long,
    median: Long,
    max: Long,
    total: Long) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("AccumProfileResults")
  }

  override def convertToSeq(): Array[String] = {
    Array(stageId.toString,
      accMetaRef.id.toString,
      accMetaRef.getName(),
      min.toString,
      median.toString,
      max.toString,
      total.toString)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(stageId.toString,
      accMetaRef.id.toString,
      accMetaRef.name.csvValue,
      min.toString,
      median.toString,
      max.toString,
      total.toString)
  }
}

case class ResourceProfileInfoCase(
    resourceProfileId: Int,
    executorResources: Map[String, ExecutorResourceRequest],
    taskResources: Map[String, TaskResourceRequest])

case class BlockManagerRemovedCase(
    executorId: String, host: String, port: Int, time: Long)

case class BlockManagerRemovedProfileResult(
    executorId: String, time: Long) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("BlockManagerRemovedProfileResult")
  }
  override def convertToSeq(): Array[String] = {
    Array(executorId, time.toString)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(executorId), time.toString)
  }
}

case class ExecutorsRemovedProfileResult(
    executorId: String, time: Long, reason: String) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("ExecutorsRemovedProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(executorId, time.toString, reason)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(executorId), time.toString,
      StringUtils.reformatCSVString(reason))
  }
}

case class UnsupportedOpsProfileResult(
    sqlID: Long, nodeID: Long, nodeName: String, nodeDescription: String,
    reason: String) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("UnsupportedOpsProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString, nodeID.toString, nodeName,
      nodeDescription, reason)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(sqlID.toString, nodeID.toString, StringUtils.reformatCSVString(nodeName),
      StringUtils.reformatCSVString(nodeDescription), StringUtils.reformatCSVString(reason))
  }
}

case class AppInfoProfileResults(
    appName: String,
    appId: Option[String],
    attemptId: Option[Int],
    sparkUser: String,
    startTime: Long,
    endTime: Option[Long],
    exitCode: Option[Int] = None,
    duration: Option[Long],
    durationStr: String,
    sparkRuntime: SparkRuntime.SparkRuntime,
    sparkVersion: String,
    pluginEnabled: Boolean,
    totalCoreSeconds: Long)  extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("AppInfoProfileResults")
  }

  private def endTimeToStr: String = {
    endTime match {
      case Some(t) => t.toString
      case None => ""
    }
  }

  private def durToStr: String = {
    duration match {
      case Some(t) => t.toString
      case None => ""
    }
  }

  private def exitCodeToStr: String = {
    exitCode.map(_.toString).getOrElse("")
  }

  private def appIdToStr: String = {
    appId match {
      case Some(t) => t
      case None => ""
    }
  }

  private def attemptIdToStr: String = {
    attemptId match {
      case Some(t) => t.toString
      case None => ""
    }
  }

  override def convertToSeq(): Array[String] = {
    Array(appName, appIdToStr, attemptIdToStr,
      sparkUser, startTime.toString, endTimeToStr, exitCodeToStr, durToStr,
      durationStr, sparkRuntime.toString, sparkVersion, pluginEnabled.toString,
      totalCoreSeconds.toString)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(appName),
      StringUtils.reformatCSVString(appIdToStr),
      StringUtils.reformatCSVString(attemptIdToStr),
      StringUtils.reformatCSVString(sparkUser),
      startTime.toString, endTimeToStr, exitCodeToStr, durToStr,
      StringUtils.reformatCSVString(durationStr),
      StringUtils.reformatCSVString(sparkRuntime.toString),
      StringUtils.reformatCSVString(sparkVersion),
      pluginEnabled.toString,
      totalCoreSeconds.toString)
  }
}

case class AppLevelRecommendationSignalsProfileResult(
    appId: String,
    numScanStagesWithGpuOom: Int,
    numGpuShuffleStagesWithContainerOom: Int) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("AppLevelRecommendationSignalsProfileResult")
  }

  override def convertToSeq(): Array[String] = Array(
    appId,
    numScanStagesWithGpuOom.toString,
    numGpuShuffleStagesWithContainerOom.toString)

  override def convertToCSVSeq(): Array[String] = Array(
    StringUtils.reformatCSVString(appId),
    numScanStagesWithGpuOom.toString,
    numGpuShuffleStagesWithContainerOom.toString)
}

object AppLevelRecommendationSignalsProfileResult {
  def build(
      appId: String,
      scanStagesWithGpuOom: Set[Long],
      gpuShuffleStagesWithContainerOom: Set[Long])
      : Seq[AppLevelRecommendationSignalsProfileResult] = {
    Seq(AppLevelRecommendationSignalsProfileResult(
      appId,
      scanStagesWithGpuOom.size,
      gpuShuffleStagesWithContainerOom.size))
  }
}

case class AppLogPathProfileResults(
    appName: String, appId: Option[String], eventLogPath: String)  extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("AppLogPathProfileResults")
  }

  override def convertToSeq(): Array[String] = {
    Array(appName, appId.getOrElse(""), eventLogPath)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(appName),
      StringUtils.reformatCSVString(appId.getOrElse("")),
      StringUtils.reformatCSVString(eventLogPath))
  }
}

case class SQLPlanMetricsCase(
    sqlID: Long,
    name: String,
    accumulatorId: Long,
    metricType: String)

case class SQLMetricInfoCase(
    sqlID: Long,
    name: String,
    accumulatorId: Long,
    metricType: String,
    nodeID: Long,
    nodeName: String,
    nodeDesc: String,
    stageIds: Set[Int])

case class DriverAccumCase(
    sqlID: Long,
    accumulatorId: Long,
    value: Long)

case class UnsupportedSQLPlan(sqlID: Long, nodeID: Long, nodeName: String,
    nodeDesc: String, reason: String)

case class FailedTaskProfileResults(
    // task info
    stageId: Int,
    stageAttemptId: Int,
    taskId: Long,
    taskAttemptId: Int,
    endReason: String,
    // used to distinguish between termination nature. If we look at taskEnd events, the values of
    // the status can be one of the following: FAILED, KILLED, SUCCESS, and UNKNOWN
    taskStatus: String,
    taskType: String,
    speculative: String,
    // task metrics
    duration: Long,
    diskBytesSpilled: Long,
    executorCPUTimeNS: Long,
    executorDeserializeCPUTimeNS: Long,
    executorDeserializeTime: Long,
    executorRunTime: Long,
    inputBytesRead: Long,
    inputRecordsRead: Long,
    jvmGCTime: Long,
    memoryBytesSpilled: Long,
    outputBytesWritten: Long,
    outputRecordsWritten: Long,
    peakExecutionMemory: Long,
    resultSerializationTime: Long,
    resultSize: Long,
    srFetchWaitTime: Long,
    srLocalBlocksFetched: Long,
    srLocalBytesRead: Long,
    srRemoteBlocksFetch: Long,
    srRemoteBytesRead: Long,
    srRemoteBytesReadToDisk: Long,
    srTotalBytesRead: Long,
    swBytesWritten: Long,
    swRecordsWritten: Long,
    swWriteTimeNS: Long) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("FailedTaskProfileResults")
  }
  private def convertToSeqInternal(reason: String): Array[String] = {
    Array(stageId.toString, stageAttemptId.toString, taskId.toString,
      taskAttemptId.toString,
      reason,
      // columns that can be used to categorize failures
      taskStatus,
      taskType,
      speculative,
      // columns for task metrics
      duration.toString,
      diskBytesSpilled.toString,
      executorCPUTimeNS.toString,
      executorDeserializeCPUTimeNS.toString,
      executorDeserializeTime.toString,
      executorRunTime.toString,
      inputBytesRead.toString,
      inputRecordsRead.toString,
      jvmGCTime.toString,
      memoryBytesSpilled.toString,
      outputBytesWritten.toString,
      outputRecordsWritten.toString,
      peakExecutionMemory.toString,
      resultSerializationTime.toString,
      resultSize.toString,
      srFetchWaitTime.toString,
      srLocalBlocksFetched.toString,
      srLocalBytesRead.toString,
      srRemoteBlocksFetch.toString,
      srRemoteBytesRead.toString,
      srRemoteBytesReadToDisk.toString,
      srTotalBytesRead.toString,
      swBytesWritten.toString,
      swRecordsWritten.toString,
      swWriteTimeNS.toString
    )
  }
  override def convertToSeq(): Array[String] = {
    convertToSeqInternal(
      // No need to escape the characters because that was done during the construction of the
      // profile result.
      StringUtils.renderStr(endReason,
        doEscapeMetaCharacters = false,
        showEllipses = true))
  }

  override def convertToCSVSeq(): Array[String] = {
    convertToSeqInternal(StringUtils.reformatCSVString(endReason))
  }
}

object FailedTaskProfileResults {
  /**
   * Constructs a FailedTaskProfileResults from a TaskModel
   * @param taskModel the taskModel to convert
   * @return a new FailedTaskProfileResults
   */
  def apply(taskModel: TaskModel): FailedTaskProfileResults = {
    FailedTaskProfileResults(
      taskModel.stageId,
      taskModel.stageAttemptId,
      taskModel.taskId,
      taskModel.attempt,
      StringUtils.renderStr(taskModel.endReason, doEscapeMetaCharacters = true, maxLength = 0),
      taskStatus = taskModel.taskStatus,
      taskModel.taskType,
      taskModel.speculative.toString,
      // fields for task metrics
      duration = taskModel.duration,
      diskBytesSpilled = taskModel.diskBytesSpilled,
      executorCPUTimeNS = taskModel.executorCPUTime,
      executorDeserializeCPUTimeNS = taskModel.executorDeserializeCPUTime,
      executorDeserializeTime = taskModel.executorDeserializeTime,
      executorRunTime = taskModel.executorRunTime,
      inputBytesRead = taskModel.input_bytesRead,
      inputRecordsRead = taskModel.input_recordsRead,
      jvmGCTime = taskModel.jvmGCTime,
      memoryBytesSpilled = taskModel.memoryBytesSpilled,
      outputBytesWritten = taskModel.output_bytesWritten,
      outputRecordsWritten = taskModel.output_recordsWritten,
      peakExecutionMemory = taskModel.peakExecutionMemory,
      resultSerializationTime = taskModel.resultSerializationTime,
      resultSize = taskModel.resultSize,
      srFetchWaitTime = taskModel.sr_fetchWaitTime,
      srLocalBlocksFetched = taskModel.sr_localBlocksFetched,
      srLocalBytesRead = taskModel.sr_localBytesRead,
      srRemoteBlocksFetch = taskModel.sr_remoteBlocksFetched,
      srRemoteBytesRead = taskModel.sr_remoteBytesRead,
      srRemoteBytesReadToDisk = taskModel.sr_remoteBytesReadToDisk,
      srTotalBytesRead = taskModel.sr_totalBytesRead,
      swBytesWritten = taskModel.sw_bytesWritten,
      swRecordsWritten = taskModel.sw_recordsWritten,
      swWriteTimeNS = taskModel.sw_writeTime
    )
  }
}

case class FailedStagesProfileResults(
    stageId: Int, stageAttemptId: Int,
    name: String, numTasks: Int, endReason: String) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("FailedStagesProfileResults")
  }
  override def convertToSeq(): Array[String] = {
    Array(stageId.toString, stageAttemptId.toString,
      name, numTasks.toString,
      StringUtils.renderStr(endReason, doEscapeMetaCharacters = true))
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(stageId.toString, stageAttemptId.toString,
      StringUtils.reformatCSVString(name), numTasks.toString,
      StringUtils.reformatCSVString(StringUtils.renderStr(endReason, doEscapeMetaCharacters = true,
        maxLength = 0)))
  }
}

case class FailedJobsProfileResults(
    jobId: Int,
    sqlID: Option[Long],  // sqlID is optional because Jobs might not have a SQL (i.e., RDDs)
    jobResult: String,
    endReason: String) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("FailedJobsProfileResults")
  }

  override def convertToSeq(): Array[String] = {
    Array(jobId.toString,
      sqlID.map(_.toString).orNull,
      jobResult,
      StringUtils.renderStr(endReason, doEscapeMetaCharacters = true))
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(jobId.toString,
      sqlID.map(_.toString).orNull,
      StringUtils.reformatCSVString(jobResult),
      StringUtils.reformatCSVString(
        StringUtils.renderStr(endReason, doEscapeMetaCharacters = true, maxLength = 0)))
  }
}

trait BaseJobStageAggTaskMetricsProfileResult extends ProfileResult {
  def id: Long
  def numTasks: Int
  def duration: Option[Long]
  def diskBytesSpilledSum: Long
  def durationSum: Long
  def durationMax: Long
  def durationMin: Long
  def durationAvg: Double
  def executorCPUTimeSum: Long // milliseconds
  def executorDeserializeCpuTimeSum: Long // milliseconds
  def executorDeserializeTimeSum: Long
  def executorRunTimeSum: Long
  def inputBytesReadSum: Long
  def inputBytesReadMax: Long
  def inputRecordsReadSum: Long
  def jvmGCTimeSum: Long
  def memoryBytesSpilledSum: Long
  def outputBytesWrittenSum: Long
  def outputRecordsWrittenSum: Long
  def peakExecutionMemoryMax: Long
  def resultSerializationTimeSum: Long
  def resultSizeMax: Long
  def srFetchWaitTimeSum: Long
  def srLocalBlocksFetchedSum: Long
  def srcLocalBytesReadSum: Long
  def srRemoteBlocksFetchSum: Long
  def srRemoteBytesReadSum: Long
  def srRemoteBytesReadToDiskSum: Long
  def srTotalBytesReadSum: Long
  def swBytesWrittenSum: Long
  def swRecordsWrittenSum: Long
  def swWriteTimeSum: Long // milliseconds

  def idHeader: String

  private def durStr: String = duration match {
    case Some(dur) => dur.toString
    case None => "null"
  }

  override def convertToSeq(): Array[String] = {
    Array(id.toString,
      numTasks.toString,
      durStr,
      diskBytesSpilledSum.toString,
      durationSum.toString,
      durationMax.toString,
      durationMin.toString,
      durationAvg.toString,
      executorCPUTimeSum.toString,
      executorDeserializeCpuTimeSum.toString,
      executorDeserializeTimeSum.toString,
      executorRunTimeSum.toString,
      inputBytesReadSum.toString,
      inputBytesReadMax.toString,
      inputRecordsReadSum.toString,
      jvmGCTimeSum.toString,
      memoryBytesSpilledSum.toString,
      outputBytesWrittenSum.toString,
      outputRecordsWrittenSum.toString,
      peakExecutionMemoryMax.toString,
      resultSerializationTimeSum.toString,
      resultSizeMax.toString,
      srFetchWaitTimeSum.toString,
      srLocalBlocksFetchedSum.toString,
      srcLocalBytesReadSum.toString,
      srRemoteBlocksFetchSum.toString,
      srRemoteBytesReadSum.toString,
      srRemoteBytesReadToDiskSum.toString,
      srTotalBytesReadSum.toString,
      swBytesWrittenSum.toString,
      swRecordsWrittenSum.toString,
      swWriteTimeSum.toString)
  }

  override def convertToCSVSeq(): Array[String] = convertToSeq()
}

case class JobAggTaskMetricsProfileResult(
    id: Long,
    numTasks: Int,
    duration: Option[Long],
    diskBytesSpilledSum: Long,
    durationSum: Long,
    durationMax: Long,
    durationMin: Long,
    durationAvg: Double,
    executorCPUTimeSum: Long, // milliseconds
    executorDeserializeCpuTimeSum: Long, // milliseconds
    executorDeserializeTimeSum: Long,
    executorRunTimeSum: Long,
    inputBytesReadSum: Long,
    inputBytesReadMax: Long,
    inputRecordsReadSum: Long,
    jvmGCTimeSum: Long,
    memoryBytesSpilledSum: Long,
    outputBytesWrittenSum: Long,
    outputRecordsWrittenSum: Long,
    peakExecutionMemoryMax: Long,
    resultSerializationTimeSum: Long,
    resultSizeMax: Long,
    srFetchWaitTimeSum: Long,
    srLocalBlocksFetchedSum: Long,
    srcLocalBytesReadSum: Long,
    srRemoteBlocksFetchSum: Long,
    srRemoteBytesReadSum: Long,
    srRemoteBytesReadToDiskSum: Long,
    srTotalBytesReadSum: Long,
    swBytesWrittenSum: Long,
    swRecordsWrittenSum: Long,
    swWriteTimeSum: Long // milliseconds
  ) extends BaseJobStageAggTaskMetricsProfileResult {
  override def idHeader = "jobId"

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("JobAggTaskMetricsProfileResult")
  }
}

case class StageAggTaskMetricsProfileResult(
    id: Long,
    numTasks: Int,
    duration: Option[Long],
    diskBytesSpilledSum: Long,
    durationSum: Long,
    durationMax: Long,
    durationMin: Long,
    durationAvg: Double,
    executorCPUTimeSum: Long, // milliseconds
    executorDeserializeCpuTimeSum: Long, // milliseconds
    executorDeserializeTimeSum: Long,
    executorRunTimeSum: Long,
    inputBytesReadSum: Long,
    inputBytesReadMax: Long,
    inputRecordsReadSum: Long,
    jvmGCTimeSum: Long,
    memoryBytesSpilledSum: Long,
    outputBytesWrittenSum: Long,
    outputRecordsWrittenSum: Long,
    peakExecutionMemoryMax: Long,
    resultSerializationTimeSum: Long,
    resultSizeMax: Long,
    srFetchWaitTimeSum: Long,
    srLocalBlocksFetchedSum: Long,
    srcLocalBytesReadSum: Long,
    srRemoteBlocksFetchSum: Long,
    srRemoteBytesReadSum: Long,
    srRemoteBytesReadToDiskSum: Long,
    srTotalBytesReadSum: Long,
    swBytesWrittenSum: Long,
    swRecordsWrittenSum: Long,
    swWriteTimeSum: Long // milliseconds
  ) extends BaseJobStageAggTaskMetricsProfileResult {

  /**
   * Combines two StageAggTaskMetricsProfileResults for the same stage.
   * This method aggregates the metrics from the current instance and the provided `other` instance.
   *
   * Detailed explanation ->
   * 1. A stage can have two successful attempts.
   * 2. We store both of those attempt information using the StageManager
   * 3. During aggregation, we combine the metrics for a stage at a stageID
   *    level
   * 4. For combining aggregated information for multiple stage attempts, we combine the
   *    aggregated per attempt information into one using the below method
   *
   * @param other The StageAggTaskMetricsProfileResult to be combined with the current instance.
   * @return A new StageAggTaskMetricsProfileResult with aggregated metrics.
   */
  def aggregateStageProfileMetric(
      other: StageAggTaskMetricsProfileResult
  ): StageAggTaskMetricsProfileResult = {
    StageAggTaskMetricsProfileResult(
      id = this.id,
      numTasks = this.numTasks + other.numTasks,
      duration = Option(this.duration.getOrElse(0L) + other.duration.getOrElse(0L)),
      diskBytesSpilledSum = this.diskBytesSpilledSum + other.diskBytesSpilledSum,
      durationSum = this.durationSum + other.durationSum,
      durationMax = Math.max(this.durationMax, other.durationMax),
      durationMin = Math.min(this.durationMin, other.durationMin),
      durationAvg = (this.durationAvg + other.durationAvg) / 2,
      executorCPUTimeSum = this.executorCPUTimeSum + other.executorCPUTimeSum,
      executorDeserializeCpuTimeSum = this.executorDeserializeCpuTimeSum +
        other.executorDeserializeCpuTimeSum,
      executorDeserializeTimeSum = this.executorDeserializeTimeSum +
        other.executorDeserializeTimeSum,
      executorRunTimeSum = this.executorRunTimeSum + other.executorRunTimeSum,
      inputBytesReadSum = this.inputBytesReadSum + other.inputBytesReadSum,
      inputBytesReadMax = Math.max(this.inputBytesReadMax, other.inputBytesReadMax),
      inputRecordsReadSum = this.inputRecordsReadSum + other.inputRecordsReadSum,
      jvmGCTimeSum = this.jvmGCTimeSum + other.jvmGCTimeSum,
      memoryBytesSpilledSum = this.memoryBytesSpilledSum + other.memoryBytesSpilledSum,
      outputBytesWrittenSum = this.outputBytesWrittenSum + other.outputBytesWrittenSum,
      outputRecordsWrittenSum = this.outputRecordsWrittenSum + other.outputRecordsWrittenSum,
      peakExecutionMemoryMax = Math.max(this.peakExecutionMemoryMax, other.peakExecutionMemoryMax),
      resultSerializationTimeSum = this.resultSerializationTimeSum +
        other.resultSerializationTimeSum,
      resultSizeMax = Math.max(this.resultSizeMax, other.resultSizeMax),
      srFetchWaitTimeSum = this.srFetchWaitTimeSum + other.srFetchWaitTimeSum,
      srLocalBlocksFetchedSum = this.srLocalBlocksFetchedSum + other.srLocalBlocksFetchedSum,
      srRemoteBlocksFetchSum = this.srRemoteBlocksFetchSum + other.srRemoteBlocksFetchSum,
      srRemoteBytesReadSum = this.srRemoteBytesReadSum + other.srRemoteBytesReadSum,
      srRemoteBytesReadToDiskSum = this.srRemoteBytesReadToDiskSum +
        other.srRemoteBytesReadToDiskSum,
      srTotalBytesReadSum = this.srTotalBytesReadSum + other.srTotalBytesReadSum,
      srcLocalBytesReadSum = this.srcLocalBytesReadSum + other.srcLocalBytesReadSum,
      swBytesWrittenSum = this.swBytesWrittenSum + other.swBytesWrittenSum,
      swRecordsWrittenSum = this.swRecordsWrittenSum + other.swRecordsWrittenSum,
      swWriteTimeSum = this.swWriteTimeSum + other.swWriteTimeSum
    )
  }

  override def idHeader = "stageId"

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("StageAggTaskMetricsProfileResult")
  }
}

/**
 * Represents diagnostic metrics results at task/stage level in a Spark SQL execution plan.
 * Output file: stage_level_diagnostic_metrics.csv.
 * Collected metrics include:
 * - Memory spilled (MB)
 * - Disk spilled (MB)
 * - Input bytes read
 * - Output bytes written
 * - Shuffle read total bytes (remote + local)
 * - Shuffle write bytes
 * - Shuffle read fetch wait time (ms)
 * - Shuffle write time (ms)
 * - GPU semaphore time (ns)
 */
case class StageDiagnosticResult(
    appName: String,
    appId: String,
    stageId: Long,
    duration: Option[Long],
    numTasks: Int,
    srTotalBytesReadMin: Long,
    srTotalBytesReadMed: Long,
    srTotalBytesReadMax: Long,
    srTotalBytesReadSum: Long,
    memoryBytesSpilled: AccumProfileResults,
    diskBytesSpilled: AccumProfileResults,
    inputBytesRead: AccumProfileResults,
    outputBytesWritten: AccumProfileResults,
    swBytesWritten: AccumProfileResults,
    srFetchWaitTime: AccumProfileResults,
    swWriteTime: AccumProfileResults,
    gpuSemaphoreWait: AccumProfileResults,
    nodeNames: Seq[String]) extends ProfileResult {

  def bytesToMB(numBytes: Long): Long = numBytes / (1024 * 1024)

  def nanoToMilliSec(numNano: Long): Long = numNano / 1000000

  private def durStr: String = duration match {
    case Some(dur) => dur.toString
    case None => "null"
  }

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("StageDiagnosticResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(appName,
      appId,
      stageId.toString,
      durStr,
      numTasks.toString,
      bytesToMB(memoryBytesSpilled.min).toString,
      bytesToMB(memoryBytesSpilled.median).toString,
      bytesToMB(memoryBytesSpilled.max).toString,
      bytesToMB(memoryBytesSpilled.total).toString,
      bytesToMB(diskBytesSpilled.min).toString,
      bytesToMB(diskBytesSpilled.median).toString,
      bytesToMB(diskBytesSpilled.max).toString,
      bytesToMB(diskBytesSpilled.total).toString,
      inputBytesRead.min.toString,
      inputBytesRead.median.toString,
      inputBytesRead.max.toString,
      inputBytesRead.total.toString,
      outputBytesWritten.min.toString,
      outputBytesWritten.median.toString,
      outputBytesWritten.max.toString,
      outputBytesWritten.total.toString,
      srTotalBytesReadMin.toString,
      srTotalBytesReadMed.toString,
      srTotalBytesReadMax.toString,
      srTotalBytesReadSum.toString,
      swBytesWritten.min.toString,
      swBytesWritten.median.toString,
      swBytesWritten.max.toString,
      swBytesWritten.total.toString,
      nanoToMilliSec(srFetchWaitTime.min).toString,
      nanoToMilliSec(srFetchWaitTime.median).toString,
      nanoToMilliSec(srFetchWaitTime.max).toString,
      nanoToMilliSec(srFetchWaitTime.total).toString,
      nanoToMilliSec(swWriteTime.min).toString,
      nanoToMilliSec(swWriteTime.median).toString,
      nanoToMilliSec(swWriteTime.max).toString,
      nanoToMilliSec(swWriteTime.total).toString,
      gpuSemaphoreWait.total.toString,
      nodeNames.mkString(","))
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(appName,
      appId,
      stageId.toString,
      durStr,
      numTasks.toString,
      bytesToMB(memoryBytesSpilled.min).toString,
      bytesToMB(memoryBytesSpilled.median).toString,
      bytesToMB(memoryBytesSpilled.max).toString,
      bytesToMB(memoryBytesSpilled.total).toString,
      bytesToMB(diskBytesSpilled.min).toString,
      bytesToMB(diskBytesSpilled.median).toString,
      bytesToMB(diskBytesSpilled.max).toString,
      bytesToMB(diskBytesSpilled.total).toString,
      inputBytesRead.min.toString,
      inputBytesRead.median.toString,
      inputBytesRead.max.toString,
      inputBytesRead.total.toString,
      outputBytesWritten.min.toString,
      outputBytesWritten.median.toString,
      outputBytesWritten.max.toString,
      outputBytesWritten.total.toString,
      srTotalBytesReadMin.toString,
      srTotalBytesReadMed.toString,
      srTotalBytesReadMax.toString,
      srTotalBytesReadSum.toString,
      swBytesWritten.min.toString,
      swBytesWritten.median.toString,
      swBytesWritten.max.toString,
      swBytesWritten.total.toString,
      nanoToMilliSec(srFetchWaitTime.min).toString,
      nanoToMilliSec(srFetchWaitTime.median).toString,
      nanoToMilliSec(srFetchWaitTime.max).toString,
      nanoToMilliSec(srFetchWaitTime.total).toString,
      nanoToMilliSec(swWriteTime.min).toString,
      nanoToMilliSec(swWriteTime.median).toString,
      nanoToMilliSec(swWriteTime.max).toString,
      nanoToMilliSec(swWriteTime.total).toString,
      gpuSemaphoreWait.total.toString,
      StringUtils.reformatCSVString(nodeNames.mkString(",")))
  }
}

case class SQLTaskAggMetricsProfileResult(
    appId: String,
    sqlId: Long,
    description: String,
    numTasks: Int,
    duration: Option[Long],
    executorCpuRatio: Double,
    diskBytesSpilledSum: Long,
    durationSum: Long,
    durationMax: Long,
    durationMin: Long,
    durationAvg: Double,
    executorCPUTimeSum: Long, // milliseconds
    executorDeserializeCpuTimeSum: Long, // milliseconds
    executorDeserializeTimeSum: Long,
    executorRunTimeSum: Long,
    inputBytesReadSum: Long,
    inputBytesReadMax: Long,
    // Not added to the output since it is used only by the AutoTuner
    inputBytesReadAvg: Double,
    inputRecordsReadSum: Long,
    jvmGCTimeSum: Long,
    memoryBytesSpilledSum: Long,
    outputBytesWrittenSum: Long,
    outputRecordsWrittenSum: Long,
    peakExecutionMemoryMax: Long,
    resultSerializationTimeSum: Long,
    resultSizeMax: Long,
    srFetchWaitTimeSum: Long,
    srLocalBlocksFetchedSum: Long,
    srcLocalBytesReadSum: Long,
    srRemoteBlocksFetchSum: Long,
    srRemoteBytesReadSum: Long,
    srRemoteBytesReadToDiskSum: Long,
    srTotalBytesReadSum: Long,
    swBytesWrittenSum: Long,
    swRecordsWrittenSum: Long,
    swWriteTimeSum: Long // milliseconds
  ) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLTaskAggMetricsProfileResult")
  }

  private def durStr: String = duration match {
    case Some(dur) => dur.toString
    case None => ""
  }

  override def convertToSeq(): Array[String] = {
    Array(appId,
      sqlId.toString,
      description,
      numTasks.toString,
      durStr,
      executorCpuRatio.toString,
      diskBytesSpilledSum.toString,
      durationSum.toString,
      durationMax.toString,
      durationMin.toString,
      durationAvg.toString,
      executorCPUTimeSum.toString,
      executorDeserializeCpuTimeSum.toString,
      executorDeserializeTimeSum.toString,
      executorRunTimeSum.toString,
      inputBytesReadSum.toString,
      inputBytesReadMax.toString,
      inputRecordsReadSum.toString,
      jvmGCTimeSum.toString,
      memoryBytesSpilledSum.toString,
      outputBytesWrittenSum.toString,
      outputRecordsWrittenSum.toString,
      peakExecutionMemoryMax.toString,
      resultSerializationTimeSum.toString,
      resultSizeMax.toString,
      srFetchWaitTimeSum.toString,
      srLocalBlocksFetchedSum.toString,
      srcLocalBytesReadSum.toString,
      srRemoteBlocksFetchSum.toString,
      srRemoteBytesReadSum.toString,
      srRemoteBytesReadToDiskSum.toString,
      srTotalBytesReadSum.toString,
      swBytesWrittenSum.toString,
      swRecordsWrittenSum.toString,
      swWriteTimeSum.toString)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(appId),
      sqlId.toString,
      StringUtils.reformatCSVString(description),
      numTasks.toString,
      durStr,
      executorCpuRatio.toString,
      diskBytesSpilledSum.toString,
      durationSum.toString,
      durationMax.toString,
      durationMin.toString,
      durationAvg.toString,
      executorCPUTimeSum.toString,
      executorDeserializeCpuTimeSum.toString,
      executorDeserializeTimeSum.toString,
      executorRunTimeSum.toString,
      inputBytesReadSum.toString,
      inputBytesReadMax.toString,
      inputRecordsReadSum.toString,
      jvmGCTimeSum.toString,
      memoryBytesSpilledSum.toString,
      outputBytesWrittenSum.toString,
      outputRecordsWrittenSum.toString,
      peakExecutionMemoryMax.toString,
      resultSerializationTimeSum.toString,
      resultSizeMax.toString,
      srFetchWaitTimeSum.toString,
      srLocalBlocksFetchedSum.toString,
      srcLocalBytesReadSum.toString,
      srRemoteBlocksFetchSum.toString,
      srRemoteBytesReadSum.toString,
      srRemoteBytesReadToDiskSum.toString,
      srTotalBytesReadSum.toString,
      swBytesWrittenSum.toString,
      swRecordsWrittenSum.toString,
      swWriteTimeSum.toString)
  }
}

/**
 * Represents IO-related diagnostic metrics results in a Spark SQL execution plan.
 * Output file: io_diagnostic_metrics.csv.
 * Collected metrics include:
 * - Output rows
 * - Scan time (ns)
 * - Output batches
 * - Buffer time (ns)
 * - Shuffle write time (ns)
 * - Fetch wait time (ns)
 * - GPU decode time (ns)
 */
case class IODiagnosticResult(
    appName: String,
    appId: String,
    sqlId: Long,
    stageId: Long,
    duration: Long,
    nodeId: Long,
    nodeName: String,
    outputRows: StatisticsMetrics,
    scanTime: StatisticsMetrics,
    outputBatches: StatisticsMetrics,
    bufferTime: StatisticsMetrics,
    shuffleWriteTime: StatisticsMetrics,
    fetchWaitTime: StatisticsMetrics,
    gpuDecodeTime: StatisticsMetrics) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("IODiagnosticResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(appName,
      appId,
      sqlId.toString,
      stageId.toString,
      duration.toString,
      nodeId.toString,
      nodeName,
      outputRows.min.toString,
      outputRows.med.toString,
      outputRows.max.toString,
      outputRows.total.toString,
      scanTime.min.toString,
      scanTime.med.toString,
      scanTime.max.toString,
      scanTime.total.toString,
      outputBatches.min.toString,
      outputBatches.med.toString,
      outputBatches.max.toString,
      outputBatches.total.toString,
      bufferTime.min.toString,
      bufferTime.med.toString,
      bufferTime.max.toString,
      bufferTime.total.toString,
      shuffleWriteTime.min.toString,
      shuffleWriteTime.med.toString,
      shuffleWriteTime.max.toString,
      shuffleWriteTime.total.toString,
      fetchWaitTime.min.toString,
      fetchWaitTime.med.toString,
      fetchWaitTime.max.toString,
      fetchWaitTime.total.toString,
      gpuDecodeTime.min.toString,
      gpuDecodeTime.med.toString,
      gpuDecodeTime.max.toString,
      gpuDecodeTime.total.toString)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(appName,
      appId,
      sqlId.toString,
      stageId.toString,
      duration.toString,
      nodeId.toString,
      StringUtils.reformatCSVString(nodeName),
      outputRows.min.toString,
      outputRows.med.toString,
      outputRows.max.toString,
      outputRows.total.toString,
      scanTime.min.toString,
      scanTime.med.toString,
      scanTime.max.toString,
      scanTime.total.toString,
      outputBatches.min.toString,
      outputBatches.med.toString,
      outputBatches.max.toString,
      outputBatches.total.toString,
      bufferTime.min.toString,
      bufferTime.med.toString,
      bufferTime.max.toString,
      bufferTime.total.toString,
      shuffleWriteTime.min.toString,
      shuffleWriteTime.med.toString,
      shuffleWriteTime.max.toString,
      shuffleWriteTime.total.toString,
      fetchWaitTime.min.toString,
      fetchWaitTime.med.toString,
      fetchWaitTime.max.toString,
      fetchWaitTime.total.toString,
      gpuDecodeTime.min.toString,
      gpuDecodeTime.med.toString,
      gpuDecodeTime.max.toString,
      gpuDecodeTime.total.toString)
  }
}

case class IOAnalysisProfileResult(
    appId: String,
    sqlId: Long,
    inputBytesReadSum: Long,
    inputRecordsReadSum: Long,
    outputBytesWrittenSum: Long,
    outputRecordsWrittenSum: Long,
    diskBytesSpilledSum: Long,
    memoryBytesSpilledSum: Long,
    srTotalBytesReadSum: Long,
    swTotalBytesWriteSum: Long) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("IOAnalysisProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(appId,
      sqlId.toString,
      inputBytesReadSum.toString,
      inputRecordsReadSum.toString,
      outputBytesWrittenSum.toString,
      outputRecordsWrittenSum.toString,
      diskBytesSpilledSum.toString,
      memoryBytesSpilledSum.toString,
      srTotalBytesReadSum.toString,
      swTotalBytesWriteSum.toString)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(appId),
      sqlId.toString,
      inputBytesReadSum.toString,
      inputRecordsReadSum.toString,
      outputBytesWrittenSum.toString,
      outputRecordsWrittenSum.toString,
      diskBytesSpilledSum.toString,
      memoryBytesSpilledSum.toString,
      srTotalBytesReadSum.toString,
      swTotalBytesWriteSum.toString)
  }
}

case class SQLDurationExecutorTimeProfileResult(
    appId: String,
    rootsqlID: Option[Long],
    sqlID: Long,
    duration: Option[Long],
    containsDataset: Boolean,
    appDuration: Option[Long],
    potentialProbs: String,
    executorCpuRatio: Double) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLDurationExecutorTimeProfileResult")
  }
  private def durStr: String = duration match {
    case Some(dur) => dur.toString
    case None => ""
  }
  private def appDurStr: String = appDuration match {
    case Some(dur) => dur.toString
    case None => ""
  }
  private def execCpuTimePercent: String = if (executorCpuRatio == -1.0) {
    "null"
  } else {
    executorCpuRatio.toString
  }
  private def potentialStr: String = if (potentialProbs.isEmpty) {
    "null"
  } else {
    potentialProbs
  }

  override def convertToSeq(): Array[String] = {
    Array(rootsqlID.getOrElse("").toString,
      appId,
      sqlID.toString,
      durStr,
      containsDataset.toString,
      appDurStr,
      potentialStr,
      execCpuTimePercent)
  }

  override def convertToCSVSeq(): Array[String] = {
    Array(StringUtils.reformatCSVString(appId),
      rootsqlID.getOrElse("").toString,
      sqlID.toString,
      durStr,
      containsDataset.toString,
      appDurStr,
      StringUtils.reformatCSVString(potentialStr),
      execCpuTimePercent)
  }
}

case class ShuffleSkewProfileResult(stageId: Long, stageAttemptId: Long,
    taskId: Long, taskAttemptId: Long, taskDuration: Long, avgDuration: Double,
    taskShuffleReadMB: Long, avgShuffleReadMB: Double, taskPeakMemoryMB: Long,
    successful: Boolean, reason: String) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("ShuffleSkewProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(stageId.toString,
      stageAttemptId.toString,
      taskId.toString,
      taskAttemptId.toString,
      f"${taskDuration.toDouble / 1000}%1.2f",
      f"${avgDuration / 1000}%1.1f",
      f"${taskShuffleReadMB.toDouble / 1024 / 1024}%1.2f",
      f"${avgShuffleReadMB / 1024 / 1024}%1.2f",
      f"${taskPeakMemoryMB.toDouble / 1024 / 1024}%1.2f",
      successful.toString,
      StringUtils.renderStr(reason, doEscapeMetaCharacters = true))
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(stageId.toString,
      stageAttemptId.toString,
      taskId.toString,
      taskAttemptId.toString,
      f"${taskDuration.toDouble / 1000}%1.2f",
      f"${avgDuration / 1000}%1.1f",
      f"${taskShuffleReadMB.toDouble / 1024 / 1024}%1.2f",
      f"${avgShuffleReadMB / 1024 / 1024}%1.2f",
      f"${taskPeakMemoryMB.toDouble / 1024 / 1024}%1.2f",
      successful.toString,
      // truncate this field because it might be a very long string, and it is already
      // fully displayed in failed_* files
      StringUtils.reformatCSVString(
        StringUtils.renderStr(reason, doEscapeMetaCharacters = true)))
  }
}

case class RapidsPropertyProfileResult(key: String, rows: Array[String]) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("RapidsPropertyProfileResult")
  }
  override def convertToSeq(): Array[String] = rows
  override def convertToCSVSeq(): Array[String] = {
    rows.map(StringUtils.reformatCSVString)
  }
}

case class WholeStageCodeGenResults(
    sqlID: Long,
    nodeID: Long,
    parent: String,
    child: String,
    childNodeID: Long) extends ProfileResult {
  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("WholeStageCodeGenResults")
  }
  override def convertToSeq(): Array[String] = {
    Array(sqlID.toString,
      nodeID.toString,
      parent,
      child,
      childNodeID.toString)
  }
  override def convertToCSVSeq(): Array[String] = {
    Array(sqlID.toString,
      nodeID.toString,
      StringUtils.reformatCSVString(parent),
      StringUtils.reformatCSVString(child),
      childNodeID.toString)
  }
}

case class RecommendedPropertyResult(property: String, value: String) {
  override def toString: String = "--conf %s=%s".format(property, value)
}

case class RecommendedCommentResult(comment: String) {
  override def toString: String = "- %s".format(comment)
}

// scalastyle:off line.size.limit
/**
 * Helper object to detect OOM exceptions from SparkRapids event logs.
 *
 * GPU OOM class names from spark-rapids-jni:
 * - GpuOOM -> GpuRetryOOM, GpuSplitAndRetryOOM
 * See: https://github.com/NVIDIA/cudf-spark-jni/blob/725cd64be2115cd072bf51d7d6c5281d6d08bf4f/src/main/cpp/src/SparkResourceAdaptorJni.cpp#L1313
 * See: https://github.com/NVIDIA/cudf-spark/blob/79922d62a1c5759963e969018322ad8e544629ff/sql-plugin/src/main/scala/com/nvidia/spark/rapids/RmmRapidsRetryIterator.scala
 */
// scalastyle:on line.size.limit
object SparkRapidsOomExceptions {
  // Current JNI: GpuOOM -> GpuRetryOOM, GpuSplitAndRetryOOM
  // Pre-24.02 JNI: jni.SplitAndRetryOOM, jni.RetryOOM (no Gpu prefix)
  // Using "jni." prefix to avoid matching CpuSplitAndRetryOOM / CpuRetryOOM
  // Using "jni.GpuOOM" (anchored) for the base class to avoid partial matches
  val gpuExceptionClassNames: Set[String] =
    Set("GpuSplitAndRetryOOM", "GpuRetryOOM", "jni.GpuOOM",
        "jni.SplitAndRetryOOM", "jni.RetryOOM")

  val gpuShuffleClassName: String = "GpuShuffleExchangeExec"

  /**
   * Check if a failure reason indicates a GPU OOM error.
   */
  def isGpuOom(failureReason: String): Boolean = {
    gpuExceptionClassNames.exists(failureReason.contains)
  }
}

/**
 * Helper object to store the exit codes for a UNIX process.
 */
object UnixExitCode {
  val FORCE_KILLED = 137
}

/**
 * GPU task metric aggregation at stage level — one row per (stageId, metricName).
 * Long/transposed schema: unit and sum/max/avg vary by metric. Empty `sum` / `avg`
 * denote max-aggregated metrics (e.g. `gpuMaxDeviceMemoryBytes`).
 *
 * Note on stage attempts: unlike StageAggTaskMetricsProfileResult, this class has
 * no aggregateStageProfileMetric helper because attempt merging happens upstream
 * at the AccumInfo layer — `AccumInfo.stagesStatMap` is keyed by stageId only
 * (not stageId + attemptNumber), so calculateAccStatsForStage already returns
 * the merged result across attempts.
 */
case class StageAggGpuMetricsProfileResult(
    stageId: Int,
    numTasks: Int,
    metricName: String,
    unit: String,
    sum: Option[Long],
    max: Option[Long],
    avg: Option[Long]) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("StageAggGpuMetricsProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(
      stageId.toString,
      numTasks.toString,
      metricName,
      unit,
      sum.map(_.toString).getOrElse(""),
      max.map(_.toString).getOrElse(""),
      avg.map(_.toString).getOrElse(""))
  }

  override def convertToCSVSeq(): Array[String] = convertToSeq()
}

/**
 * GPU task metric aggregation at SQL level — one row per (sqlId, metricName).
 * Rolled up from stage-level rows: sum = Σ stage.sum, max = max stage.max,
 * avg = task-weighted average over stage.avg. numTasks is intentionally not
 * carried — it would be a constant per SQL across every metric row (the non-GPU
 * sql_level_aggregated_task_metrics.csv already has it once per SQL).
 */
case class SQLAggGpuMetricsProfileResult(
    sqlId: Long,
    metricName: String,
    unit: String,
    sum: Option[Long],
    max: Option[Long],
    avg: Option[Long]) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("SQLAggGpuMetricsProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(
      sqlId.toString,
      metricName,
      unit,
      sum.map(_.toString).getOrElse(""),
      max.map(_.toString).getOrElse(""),
      avg.map(_.toString).getOrElse(""))
  }

  override def convertToCSVSeq(): Array[String] = convertToSeq()
}

/**
 * GPU task metric aggregation at app level — one row per (appId, metricName).
 * Same rollup rules as SQL-level. numTasks intentionally omitted (would be a
 * constant per app and is available in existing per-app CSVs).
 */
case class AppAggGpuMetricsProfileResult(
    appId: String,
    metricName: String,
    unit: String,
    sum: Option[Long],
    max: Option[Long],
    avg: Option[Long]) extends ProfileResult {

  override def outputHeaders: Array[String] = {
    OutHeaderRegistry.outputHeaders("AppAggGpuMetricsProfileResult")
  }

  override def convertToSeq(): Array[String] = {
    Array(
      appId,
      metricName,
      unit,
      sum.map(_.toString).getOrElse(""),
      max.map(_.toString).getOrElse(""),
      avg.map(_.toString).getOrElse(""))
  }

  override def convertToCSVSeq(): Array[String] = convertToSeq()
}
