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

object StageAccumDiagnosticMetrics {
  val MEMORY_SPILLED_METRIC = "internal.metrics.memoryBytesSpilled"
  val DISK_SPILLED_METRIC = "internal.metrics.diskBytesSpilled"
  val INPUT_BYTES_READ_METRIC = "internal.metrics.input.bytesRead"
  val OUTPUT_BYTES_WRITTEN_METRIC = "internal.metrics.output.bytesWritten"
  val SW_TOTAL_BYTES_METRIC = "internal.metrics.shuffle.write.bytesWritten"
  val SR_FETCH_WAIT_TIME_METRIC = "internal.metrics.shuffle.read.fetchWaitTime"
  val SW_WRITE_TIME_METRIC = "internal.metrics.shuffle.write.writeTime"
  val GPU_SEMAPHORE_WAIT_METRIC = "gpuSemaphoreWait"

  /**
   * Set of all diagnostic metrics
   */
  lazy val allDiagnosticMetrics: Set[String] = Set(MEMORY_SPILLED_METRIC,
    DISK_SPILLED_METRIC, INPUT_BYTES_READ_METRIC, OUTPUT_BYTES_WRITTEN_METRIC,
    SW_TOTAL_BYTES_METRIC, SR_FETCH_WAIT_TIME_METRIC, SW_WRITE_TIME_METRIC,
    GPU_SEMAPHORE_WAIT_METRIC)

  /**
   * Check if a metric is diagnostic
   */
  def isDiagnosticMetrics(metric: String): Boolean = {
    allDiagnosticMetrics.contains(metric)
  }
}

object IOAccumDiagnosticMetrics {
  // Metric keys to support variations in metric naming
  val OUTPUT_ROWS_METRIC_KEY = "output rows"
  val SCAN_TIME_METRIC_KEY = "scan time"
  val OUTPUT_BATCHES_METRIC_KEY = "output batches"
  val BUFFER_TIME_METRIC_KEY = "buffer time"
  val SHUFFLE_WRITE_TIME_METRIC_KEY = "shuffle write time"
  val FETCH_WAIT_TIME_METRIC_KEY = "fetch wait time"
  val GPU_DECODE_TIME_METRIC_KEY = "GPU decode time"
  val DECODED_BATCH_BYTES_METRIC_KEY = "decoded batch bytes"

  val OUTPUT_ROW_METRIC_NAMES = Set(
    "number of output rows", // common across all Spark eventlogs
    "output rows", // only in GPU eventlogs
    "join output rows" // only in GPU eventlogs
  )

  val SCAN_TIME_METRIC_NAMES = Set(
    "scan time" // common across all Spark eventlogs
  )

  val OUTPUT_BATCHES_METRIC_NAMES = Set(
    "number of output batches", // only in Photon eventlogs
    "output columnar batches" // only in GPU eventlogs
  )

  val BUFFER_TIME_METRIC_NAMES = Set(
    "buffer time" // common across all Spark eventlogs
  )

  val SHUFFLE_WRITE_TIME_METRIC_NAMES = Set(
    "shuffle write time", // common across all Spark eventlogs
    "rs. shuffle write time" // only in GPU eventlogs
  )

  val FETCH_WAIT_TIME_METRIC_NAMES = Set(
    "fetch wait time" // common across all Spark eventlogs
  )

  val GPU_DECODE_TIME_METRIC_NAMES = Set(
    "GPU decode time" // only in GPU eventlogs
  )

  val DECODED_BATCH_BYTES_METRIC_NAMES = Set(
    "decoded batch bytes" // only in GPU eventlogs, on scan nodes
  )

  private val metricNamesToKeyMap: Map[String, String] = (
    OUTPUT_ROW_METRIC_NAMES.map(_ -> OUTPUT_ROWS_METRIC_KEY) ++
    SCAN_TIME_METRIC_NAMES.map(_ -> SCAN_TIME_METRIC_KEY) ++
    OUTPUT_BATCHES_METRIC_NAMES.map(_ -> OUTPUT_BATCHES_METRIC_KEY) ++
    BUFFER_TIME_METRIC_NAMES.map(_ -> BUFFER_TIME_METRIC_KEY) ++
    SHUFFLE_WRITE_TIME_METRIC_NAMES.map(_ -> SHUFFLE_WRITE_TIME_METRIC_KEY) ++
    FETCH_WAIT_TIME_METRIC_NAMES.map(_ -> FETCH_WAIT_TIME_METRIC_KEY) ++
    GPU_DECODE_TIME_METRIC_NAMES.map(_ -> GPU_DECODE_TIME_METRIC_KEY) ++
    DECODED_BATCH_BYTES_METRIC_NAMES.map(_ -> DECODED_BATCH_BYTES_METRIC_KEY)).toMap

  /**
   * Set of all IO diagnostic metrics names
   */
  lazy val allIODiagnosticMetrics: Set[String] = metricNamesToKeyMap.keys.toSet

  /**
   * Check if a metric name belongs to IO diagnostic metrics
   */
  def isIODiagnosticMetricName(metric: String): Boolean = {
    allIODiagnosticMetrics.contains(metric)
  }

  /**
   * Normalize a metric name to its IO diagnostic metric constant because we want to
   * support variations in metric naming, e.g. "join output rows", "number of output rows"
   * are different names for output rows metric.
   */
  def normalizeToIODiagnosticMetricKey(metric: String): String = {
    // input metric is already known to be an IO diagnostic metric
    metricNamesToKeyMap(metric)
  }
}
