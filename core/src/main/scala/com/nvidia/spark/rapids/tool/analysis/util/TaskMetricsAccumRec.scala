/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.analysis.util

import com.nvidia.spark.rapids.tool.profiling.StageAggTaskMetricsProfileResult

import org.apache.spark.sql.rapids.tool.ToolUtils
import org.apache.spark.sql.rapids.tool.store.TaskModel

/**
 * Accumulator used for task metrics.
 * This is an optimization decision to avoid using Scala builtin collections on every field in the
 * taskModel.
 */
class TaskMetricsAccumRec {
  var numTasks: Int = 0
  var diskBytesSpilledSum: Long = 0
  var durationSum: Long = 0
  var durationMax: Long = Long.MinValue
  var durationMin: Long = Long.MaxValue
  var durationAvg: Double = 0.0
  var executorCPUTimeSum: Long = 0
  var executorDeserializeCpuTimeSum: Long = 0
  var executorDeserializeTimeSum: Long = 0
  var executorRunTimeSum: Long = 0
  var inputBytesReadSum: Long = 0
  var inputBytesReadMax: Long = Long.MinValue
  var inputRecordsReadSum: Long = 0
  var jvmGCTimeSum: Long = 0
  var memoryBytesSpilledSum: Long = 0
  var outputBytesWrittenSum: Long = 0
  var outputRecordsWrittenSum: Long = 0
  var peakExecutionMemoryMax: Long = Long.MinValue
  var resultSerializationTimeSum: Long = 0
  var resultSizeMax: Long = Long.MinValue
  var srFetchWaitTimeSum: Long = 0
  var srLocalBlocksFetchedSum: Long = 0
  var srLocalBytesReadSum: Long = 0
  var srRemoteBlocksFetchSum: Long = 0
  var srRemoteBytesReadSum: Long = 0
  var srRemoteBytesReadToDiskSum: Long = 0
  var srTotalBytesReadSum: Long = 0
  var swBytesWrittenSum: Long = 0
  var swRecordsWrittenSum: Long = 0
  var swWriteTimeSum: Long = 0

  /**
   * Assumption that 0-tasks implies no aggregations on metrics. This means that metrics on
   * job/SQL levels won't be accumulated as long as no tasks are accounted for.
   */
  def isEmptyAggregates: Boolean = numTasks == 0

  /**
   * Reset all fields to 0. This is used to reset the fields when the Task iterator is empty.
   * When the iterator is empty, then fields such as "max" should be reset to 0.
   */
  def resetFields(): Unit = {
    durationMax = 0
    durationMin = 0
    inputBytesReadMax = 0
    peakExecutionMemoryMax = 0
    resultSizeMax = 0
  }

  def addRecord(rec: TaskModel): Unit = {
    numTasks += 1
    // SumFields
    diskBytesSpilledSum += rec.diskBytesSpilled
    durationSum += rec.duration
    executorCPUTimeSum += rec.executorCPUTime
    executorDeserializeCpuTimeSum += rec.executorDeserializeCPUTime
    executorDeserializeTimeSum += rec.executorDeserializeTime
    executorRunTimeSum += rec.executorRunTime
    inputBytesReadSum += rec.input_bytesRead
    inputRecordsReadSum += rec.input_recordsRead
    jvmGCTimeSum += rec.jvmGCTime
    memoryBytesSpilledSum += rec.memoryBytesSpilled
    outputBytesWrittenSum += rec.output_bytesWritten
    outputRecordsWrittenSum += rec.output_recordsWritten
    resultSerializationTimeSum += rec.resultSerializationTime
    srFetchWaitTimeSum += rec.sr_fetchWaitTime
    srLocalBlocksFetchedSum += rec.sr_localBlocksFetched
    srLocalBytesReadSum += rec.sr_localBytesRead
    srRemoteBlocksFetchSum += rec.sr_remoteBlocksFetched
    srRemoteBytesReadSum += rec.sr_remoteBytesRead
    srRemoteBytesReadToDiskSum += rec.sr_remoteBytesReadToDisk
    srTotalBytesReadSum += rec.sr_totalBytesRead
    swBytesWrittenSum += rec.sw_bytesWritten
    swRecordsWrittenSum += rec.sw_recordsWritten
    swWriteTimeSum += rec.sw_writeTime
    // Max fields
    durationMax = math.max(durationMax, rec.duration)
    inputBytesReadMax = math.max(inputBytesReadMax, rec.input_bytesRead)
    peakExecutionMemoryMax = math.max(peakExecutionMemoryMax, rec.peakExecutionMemory)
    resultSizeMax = math.max(resultSizeMax, rec.resultSize)
    // Min Fields
    durationMin = math.min(durationMin, rec.duration)
  }

  def addRecord(rec: StageAggTaskMetricsProfileResult): Unit = {
    // Sums
    numTasks += rec.numTasks
    durationSum += rec.durationSum
    diskBytesSpilledSum += rec.diskBytesSpilledSum
    executorCPUTimeSum += rec.executorCPUTimeSum
    executorRunTimeSum += rec.executorRunTimeSum
    inputBytesReadSum += rec.inputBytesReadSum
    executorDeserializeCpuTimeSum += rec.executorDeserializeCpuTimeSum
    executorDeserializeTimeSum += rec.executorDeserializeTimeSum
    inputRecordsReadSum += rec.inputRecordsReadSum
    jvmGCTimeSum += rec.jvmGCTimeSum
    memoryBytesSpilledSum += rec.memoryBytesSpilledSum
    outputBytesWrittenSum += rec.outputBytesWrittenSum
    outputRecordsWrittenSum += rec.outputRecordsWrittenSum
    resultSerializationTimeSum += rec.resultSerializationTimeSum
    srFetchWaitTimeSum += rec.srFetchWaitTimeSum
    srLocalBlocksFetchedSum += rec.srLocalBlocksFetchedSum
    srLocalBytesReadSum += rec.srcLocalBytesReadSum
    srRemoteBlocksFetchSum += rec.srRemoteBlocksFetchSum
    srRemoteBytesReadSum += rec.srRemoteBytesReadSum
    srRemoteBytesReadToDiskSum += rec.srRemoteBytesReadToDiskSum
    srTotalBytesReadSum += rec.srTotalBytesReadSum
    swBytesWrittenSum += rec.swBytesWrittenSum
    swRecordsWrittenSum += rec.swRecordsWrittenSum
    swWriteTimeSum += rec.swWriteTimeSum
    // Max
    durationMax = math.max(durationMax, rec.durationMax)
    inputBytesReadMax = math.max(inputBytesReadMax, rec.inputBytesReadMax)
    peakExecutionMemoryMax = math.max(peakExecutionMemoryMax, rec.peakExecutionMemoryMax)
    resultSizeMax = math.max(resultSizeMax, rec.resultSizeMax)
    // Min. A record with no tasks carries durationMin = 0 from resetFields, which is a
    // placeholder rather than a measurement. durationMin seeds at Long.MaxValue, the identity
    // for min, so an empty accumulator needs no separate case here.
    if (rec.numTasks > 0) {
      durationMin = math.min(durationMin, rec.durationMin)
    }
  }

  /**
   * This method should be called to finalize the accumulations of all the metrics.
   * For example, calculating averages and doing any last transformations on a field before the
   * results are consumed.
   */
  def finalizeAggregation(): Unit = {
    durationAvg = ToolUtils.calculateAverage(durationSum, numTasks, 1)
    if (numTasks < 1) {
      // number of tasks is 0, then we should reset fields such as (max, min) to 0.
      resetFields()
    }
  }
}
