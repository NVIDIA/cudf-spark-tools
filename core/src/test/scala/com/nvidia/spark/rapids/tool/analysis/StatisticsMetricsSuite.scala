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


package com.nvidia.spark.rapids.tool.analysis

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.rapids.tool.store.{AccumInfo, AccumInfoWithMaxAgg, AccumMetaRef}

class StatisticsMetricsSuite extends AnyFunSuite {

  /** Two-pass reference standard deviation, computed apart from the incremental implementation. */
  private def referenceStddev(values: Seq[Long]): Double = {
    val mean = values.map(_.toDouble).sum / values.size
    val ss = values.map(v => (v - mean) * (v - mean)).sum
    Math.sqrt(ss / (values.size - 1))
  }

  private def stageEvent(info: AccumInfo, stageId: Int, value: Long): Unit = {
    info.addAccumToStage(stageId,
      org.apache.spark.scheduler.AccumulableInfo(
        1L, Some("gpuTime"), None, Some(value.toString), true, true, None))
  }

  private def feed(info: AccumInfo, stageId: Int, values: Seq[Long]): Unit = {
    values.foreach { v =>
      info.addAccumToTask(stageId,
        org.apache.spark.scheduler.AccumulableInfo(
          1L, Some("gpuTime"), Some(v.toString), None, true, true, None))
    }
  }

  test("stddev from a single stage matches a two-pass reference") {
    val values = Seq(3L, 1L, 4L, 1L, 5L, 9L, 2L, 6L)
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    feed(info, 1, values)
    val stats = info.getRawStatsForStage(1).get
    assert(stats.count == values.size)
    assert(stats.total == values.sum)
    val expected = referenceStddev(values)
    assert(stats.stddev.exists(s => Math.abs(s - expected) < 1e-9),
      s"stddev ${stats.stddev} != reference $expected")
  }

  test("stddev merged across stages matches the reference over the pooled samples") {
    // Chan's correction is what a naive sum of the two deviation sums misses, so the stage
    // means are deliberately far apart.
    val stageA = Seq(1L, 2L, 3L)
    val stageB = Seq(1000L, 1010L, 1020L, 1030L)
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    feed(info, 1, stageA)
    feed(info, 2, stageB)
    val merged = info.calculateAccStats()
    assert(merged.count == stageA.size + stageB.size)
    val expected = referenceStddev(stageA ++ stageB)
    assert(merged.stddev.exists(s => Math.abs(s - expected) < 1e-6),
      s"merged stddev ${merged.stddev} != reference $expected")
    // a naive sum of the two deviation sums would land far below the pooled spread
    assert(merged.stddev.exists(_ > 100.0), "merge dropped the between-stage variance")
  }

  test("stddev is undefined below two samples") {
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    feed(info, 1, Seq(42L))
    assert(info.getRawStatsForStage(1).get.stddev.isEmpty)
    assert(StatisticsMetrics.ZERO_RECORD.stddev.isEmpty)
  }

  test("createFromArr computes the same deviation sum as the incremental path") {
    val values = Array(7L, 11L, 13L, 17L, 19L)
    val fromArr = StatisticsMetrics.createFromArr(values.clone())
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    feed(info, 1, values.toSeq)
    val incremental = info.getRawStatsForStage(1).get
    assert(Math.abs(fromArr.stddev.get - incremental.stddev.get) < 1e-9)
    assert(Math.abs(fromArr.stddev.get - referenceStddev(values.toSeq)) < 1e-9)
  }

  test("a stage completion value does not disturb the task samples it arrives after") {
    val values = Seq(10L, 20L)
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    feed(info, 1, values)
    stageEvent(info, 1, 100L)
    val stats = info.getRawStatsForStage(1).get
    assert(stats.count == 2L)
    assert(stats.sampleTotal == 30L, "task samples must stay their own sum")
    assert(stats.min == 10L && stats.max == 20L, "min and max describe the task samples")
    assert(stats.total == 100L, "the stage value is what gets published")
    assert(stats.stddev.exists(v => Math.abs(v - referenceStddev(values)) < 1e-9))
  }

  test("a stage completion value arriving first does not reach the task samples") {
    // Event logs from some platforms deliver stage completion out of order. The published total
    // absorbs the stage value either way, which is long standing behaviour; what must not happen
    // is the samples inheriting it, because count and welfordSumSqDev do not.
    val values = Seq(10L, 20L)
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    stageEvent(info, 1, 100L)
    feed(info, 1, values)
    val stats = info.getRawStatsForStage(1).get
    assert(stats.count == 2L)
    assert(stats.sampleTotal == 30L, "the stage value must not seed the sample total")
    // The record a stage event creates holds placeholder zeros. Folding the first task update
    // against them would publish a minimum of 0 that no task produced.
    assert(stats.min == 10L, "the stage seed must not become the sample minimum")
    assert(stats.max == 20L, "the stage seed must not become the sample maximum")
    assert(stats.stddev.exists(v => Math.abs(v - referenceStddev(values)) < 1e-9))
  }

  test("a stage that reported only at completion carries a total but no samples") {
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    stageEvent(info, 1, 100L)
    val stats = info.getRawStatsForStage(1).get
    assert(stats.count == 0L)
    assert(stats.sampleTotal == 0L)
    assert(stats.total == 100L)
    assert(stats.stddev.isEmpty)
    // min and max hold placeholder zeros here, which no task produced. Readers of the sampled
    // extrema must see nothing rather than a zero.
    assert(stats.sampleMin.isEmpty && stats.sampleMax.isEmpty)
  }

  test("sampled extrema appear once a task reports and not before") {
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    stageEvent(info, 1, 100L)
    assert(info.getRawStatsForStage(1).get.sampleMax.isEmpty)
    feed(info, 1, Seq(10L, 20L))
    val stats = info.getRawStatsForStage(1).get
    assert(stats.sampleMin.contains(10L) && stats.sampleMax.contains(20L))
  }

  test("the max-aggregated record forwards the samples while replacing the total") {
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuMaxTaskFootprint")))
    feed(info, 1, Seq(10L, 20L))
    feed(info, 2, Seq(30L, 40L))
    assert(info.isInstanceOf[AccumInfoWithMaxAgg], "precondition: this metric aggregates by max")
    val merged = info.calculateAccStats()
    assert(merged.total == 40L, "a max-aggregated total is the peak, not the sum")
    assert(merged.count == 4L && merged.sampleTotal == 100L, "the samples are forwarded")
    assert(merged.min == 10L && merged.max == 40L, "the extrema are forwarded")
    val expected = referenceStddev(Seq(10L, 20L, 30L, 40L))
    assert(merged.stddev.exists(v => Math.abs(v - expected) < 1e-9),
      s"merged stddev ${merged.stddev} != $expected")
  }

  test("a lower stage completion value never lowers the published total") {
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    stageEvent(info, 1, 100L)
    stageEvent(info, 1, 40L)
    assert(info.getRawStatsForStage(1).get.total == 100L)
  }

  test("merging carries the stage totals and the task samples separately") {
    val info = AccumInfo(AccumMetaRef(1L, Some("gpuTime")))
    feed(info, 1, Seq(10L, 20L))
    stageEvent(info, 1, 100L)
    feed(info, 2, Seq(30L, 40L))
    stageEvent(info, 2, 200L)
    val merged = info.calculateAccStats()
    assert(merged.count == 4L)
    assert(merged.sampleTotal == 100L, "pooled task samples")
    assert(merged.total == 300L, "pooled published totals")
    val expected = referenceStddev(Seq(10L, 20L, 30L, 40L))
    assert(merged.stddev.exists(v => Math.abs(v - expected) < 1e-9),
      s"merged stddev ${merged.stddev} != $expected")
  }

  test("the report path refuses a standard deviation below two samples") {
    // Emitted rows call MetricCatalog.stddevOf, a different entry point from the record's own
    // stddev. A one sample row here divides by zero and renders NaN, which the CSV formatter
    // cannot represent.
    assert(MetricCatalog.DEFAULT.stddevOf("gpuTime", 0.0, 1L).isEmpty)
    assert(MetricCatalog.DEFAULT.stddevOf("gpuTime", 0.0, 0L).isEmpty)
    // a deviation sum of 25 over two samples is a variance of 25 and a stddev of 5
    val plain = MetricCatalog.DEFAULT.stddevOf("gpuTime", 25.0, 2L)
    assert(plain.exists(v => Math.abs(v - 5.0) < 1e-9))
    // and the scaled metric divides the stored scale out of the result
    val scaled = MetricCatalog.DEFAULT.stddevOf("gpuOnGpuTasksWaitingGPUAvgCount", 25.0, 2L)
    assert(scaled.exists(v => Math.abs(v - 0.005) < 1e-9))
  }
}
