/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

import com.nvidia.spark.rapids.tool.profiling.SQLAccumProfileResults
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.rapids.tool.UnsupportedMetricNameException

/**
 * Unit tests for I/O-metric recognition and routing, covering the Iceberg V2
 * `total data file size (bytes)` scan metric (issue #2115) alongside the existing
 * file-source `size of files read` metric.
 */
class IoMetricsSuite extends AnyFunSuite {

  // Iceberg BatchScan emits its scan bytes under this SQL-plan metric name.
  private val icebergDataSizeName = "total data file size (bytes)"
  private val icebergMetricType =
    "v2Custom_org.apache.iceberg.spark.source.metrics.TotalDataFileSize"

  private def accum(
      name: String,
      total: Long,
      metricType: String = "sum"): SQLAccumProfileResults = {
    SQLAccumProfileResults(
      sqlID = 0L,
      nodeID = 1L,
      nodeName = "BatchScan",
      accumulatorId = 486L,
      name = name,
      min = 0L,
      median = total,
      max = total,
      total = total,
      metricType = metricType,
      stageIds = Set(0))
  }

  test("Iceberg total data file size is recognized as an I/O metric") {
    val srcAccum = accum(icebergDataSizeName, 5177503442L, icebergMetricType)
    assert(IoMetrics.isIoMetric(icebergDataSizeName))
    assert(IoMetrics.isIoMetric(srcAccum))
  }

  test("Iceberg total data file size populates dataSize") {
    val srcAccum = accum(icebergDataSizeName, 5177503442L, icebergMetricType)
    val rec = IoMetrics(0, 0, 0, 0)
    IoMetrics.updateIoRecord(rec, srcAccum)
    assert(rec.dataSize == 5177503442L)
    // Only the data-size field is touched.
    assert(rec.bufferTime == 0 && rec.scanTime == 0 && rec.decodeTime == 0)
  }

  test("file-source 'size of files read' still routes to dataSize (regression)") {
    val srcAccum = accum("size of files read", 12345L)
    assert(IoMetrics.isIoMetric(srcAccum))
    val rec = IoMetrics(0, 0, 0, 0)
    IoMetrics.updateIoRecord(rec, srcAccum)
    assert(rec.dataSize == 12345L)
  }

  test("Photon and Auron helpers also recognize and route the Iceberg metric") {
    for (helper <- Seq(PhotonIoMetrics, AuronIoMetrics)) {
      val srcAccum = accum(icebergDataSizeName, 987654321L, icebergMetricType)
      assert(helper.isIoMetric(srcAccum), s"$helper should recognize the Iceberg metric")
      val rec = IoMetrics(0, 0, 0, 0)
      helper.updateIoRecord(rec, srcAccum)
      assert(rec.dataSize == 987654321L, s"$helper should route the Iceberg metric to dataSize")
    }
  }

  test("unrelated metric names remain unsupported") {
    val srcAccum = accum("some unrelated metric", 1L)
    assert(!IoMetrics.isIoMetric(srcAccum))
    val rec = IoMetrics(0, 0, 0, 0)
    assertThrows[UnsupportedMetricNameException] {
      IoMetrics.updateIoRecord(rec, srcAccum)
    }
  }
}
