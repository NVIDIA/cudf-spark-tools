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

import java.util.concurrent.TimeUnit

import com.nvidia.spark.rapids.tool.planparser.db.PhotonParseHelper
import com.nvidia.spark.rapids.tool.profiling.{AccumProfileResults, SQLAccumProfileResults}

import org.apache.spark.sql.rapids.tool.{AppBase, UnsupportedMetricNameException}

/**
 * Container for I/O metrics from data source operations.
 */
case class IoMetrics(
    var bufferTime: Long,
    var scanTime: Long,
    var dataSize: Long,
    var decodeTime: Long
)

/**
 * Base trait for handling I/O metrics from different execution engines.
 */
trait IoMetricsTrait {

  protected val BUFFER_TIME_LABEL: String
  protected val SCAN_TIME_LABEL: String
  protected val DATA_SIZE_LABEL: String
  protected val DECODE_TIME_LABEL: String

  // Data-size metric emitted by Iceberg DataSource V2 BatchScan
  // (v2Custom_org.apache.iceberg.spark.source.metrics.TotalDataFileSize). It is produced by the
  // Iceberg source independent of the compute engine, so it is recognized at the base level and
  // routed to dataSize for all engine helpers (OSS, Photon, Auron).
  protected val ICEBERG_DATA_SIZE_LABEL = "total data file size (bytes)"

  protected val ioLabels: Set[String]

  /** Check if a metric name is an I/O metric */
  def isIoMetric(arg: String): Boolean = {
    ioLabels.contains(arg) || arg == ICEBERG_DATA_SIZE_LABEL
  }

  /** Check if a SQL accumulator is an I/O metric */
  def isIoMetric(srcAccum: SQLAccumProfileResults): Boolean = {
    isIoMetric(srcAccum.name)
  }

  /** Check if a metric name is a scan time metric */
  def isScanTimeMetric(accumName: String): Boolean = {
    accumName == SCAN_TIME_LABEL
  }

  /** Check if a SQL accumulator is a scan time metric */
  def isScanTimeMetric(srcAccum: SQLAccumProfileResults): Boolean = {
    isScanTimeMetric(srcAccum.name)
  }

  /** Check if an accumulator is a scan time metric */
  def isScanTimeMetric(srcAccum: AccumProfileResults): Boolean = {
    isScanTimeMetric(srcAccum.accMetaRef.getName())
  }

  /** Convert timing metrics to milliseconds */
  def convertMsTime(srcAccum: SQLAccumProfileResults): Long = {
    // Convert nsTiming to milliseconds
    srcAccum.metricType match {
      case "nsTiming" => TimeUnit.NANOSECONDS.toMillis(srcAccum.total)
      case _ => srcAccum.total
    }
  }

  /** Update IoMetrics record with accumulator data */
  def updateIoRecord(destRec: IoMetrics, srcAccum: SQLAccumProfileResults): Unit = {
    srcAccum.name match {
      case BUFFER_TIME_LABEL => destRec.bufferTime = convertMsTime(srcAccum)
      case SCAN_TIME_LABEL => destRec.scanTime = convertMsTime(srcAccum)
      case DECODE_TIME_LABEL => destRec.decodeTime = convertMsTime(srcAccum)
      case DATA_SIZE_LABEL | ICEBERG_DATA_SIZE_LABEL => destRec.dataSize = srcAccum.total
      case _ => throw UnsupportedMetricNameException(srcAccum.name)
    }
  }
}

/**
 * I/O metrics trait for open-source Spark engines.
 */
trait OssIoMetricsTrait extends IoMetricsTrait {
  override protected val BUFFER_TIME_LABEL = "buffer time"
  override protected val SCAN_TIME_LABEL = "scan time"
  override protected val DATA_SIZE_LABEL = "size of files read"
  // It is incorrect to define GPU decode time as an I/O metric, but it is included here to be
  // backward compatible with previous code that used to collect it as part of I/O metrics.
  // The correct way is to define GPU decode time in a child trait for RAPIDS engine metrics.
  override protected val DECODE_TIME_LABEL = "GPU decode time"

  override protected val ioLabels: Set[String] = Set(
    BUFFER_TIME_LABEL, SCAN_TIME_LABEL, DATA_SIZE_LABEL, DECODE_TIME_LABEL
  )
}

/**
 * Factory and utilities for I/O metrics handling.
 */
object IoMetrics extends OssIoMetricsTrait {
  val EMPTY_IO_METRICS: IoMetrics = IoMetrics(0, 0, 0, 0)

  /** Get the appropriate I/O metrics helper based on the application type */
  def getIoMetricsHelper(app: AppBase): IoMetricsTrait = {
    if (app.isAuronEnabled) {
      AuronIoMetrics
    } else if (app.dbPlugin.isPhotonEnabled) {
      PhotonIoMetrics
    } else {
      IoMetrics
    }
  }
}

/**
 * I/O metrics handler for Photon engine.
 */
object PhotonIoMetrics extends OssIoMetricsTrait {

  override def isIoMetric(srcAccum: SQLAccumProfileResults): Boolean = {
    PhotonParseHelper.isPhotonIoMetric(srcAccum) || super.isIoMetric(srcAccum)
  }

  override def isScanTimeMetric(srcAccum: SQLAccumProfileResults): Boolean = {
    PhotonParseHelper.isPhotonIoMetric(srcAccum) || super.isScanTimeMetric(srcAccum)
  }

  override def updateIoRecord(destRec: IoMetrics, srcAccum: SQLAccumProfileResults): Unit = {
    srcAccum.name match {
      case _ if PhotonParseHelper.isPhotonIoMetric(srcAccum) =>
        PhotonParseHelper.updatePhotonIoMetric(srcAccum, destRec)
      case _ => super.updateIoRecord(destRec, srcAccum)
    }
  }
}

/**
 * I/O metrics handler for Auron native engine.
 */
object AuronIoMetrics extends OssIoMetricsTrait {
  // Native.io_time: Total Native I/O Time. The cumulative time spent by the native engine
  //                 interacting with the storage layer. This is the "high-level" I/O metric.
  // Native.io_time_getfs: Filesystem Interaction Time. A subset of the I/O time specifically spent
  //                 waiting for the filesystem (HDFS, S3, etc.) to respond or provide a file
  //                 handle/stream.
  //                 Native.io_time_getfs is usually a component of Native.io_time
  // Native.elapsed_compute: Pure Processing Time: The time spent on CPU-bound tasks after the data
  //                 is in memory, such as decompression, decoding columnar data, and applying
  //                 filters (push-down).
  // The total duration of the scan operator should be Max(I/O Time, Compute Time) + Overhead
  private val AURON_SCAN_TIME_LABEL = "Native.io_time"
  private val AURON_DECODE_TIME_LABEL = "Native.elapsed_compute"
  private val AURON_DATA_SIZE_LABEL = "Native.bytes_scanned"

  override protected val ioLabels: Set[String] = Set(
      AURON_SCAN_TIME_LABEL,
      AURON_DECODE_TIME_LABEL,
      AURON_DATA_SIZE_LABEL,
      SCAN_TIME_LABEL,
      DECODE_TIME_LABEL,
      DATA_SIZE_LABEL,
      BUFFER_TIME_LABEL
    )

  override def isScanTimeMetric(accumName: String): Boolean = {
    accumName == AURON_SCAN_TIME_LABEL || super.isScanTimeMetric(accumName)
  }

  override def updateIoRecord(destRec: IoMetrics, srcAccum: SQLAccumProfileResults): Unit = {
    srcAccum.name match {
      case AURON_SCAN_TIME_LABEL =>
        destRec.scanTime = convertMsTime(srcAccum)
      case AURON_DATA_SIZE_LABEL =>
        destRec.dataSize = srcAccum.total
      case AURON_DECODE_TIME_LABEL =>
        destRec.decodeTime = convertMsTime(srcAccum)
      case _ => super.updateIoRecord(destRec, srcAccum)
    }
  }
}
