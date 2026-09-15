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

package com.nvidia.spark.rapids.tool.analysis

import org.apache.spark.sql.rapids.tool.util.InPlaceMedianArrView.{chooseMidpointPivotInPlace, findMedianInPlace}

/**
 * Store (min, median, max, count, total, sampleTotal) for a given metric, plus the sum of
 * squared deviations from the mean, known as Welford's M2. That term makes a standard
 * deviation available without keeping a value per task: it updates in constant time per
 * sample and merges across stages by Chan's parallel form. It is held in the metric's
 * fixed-point storage units, so a variance carries the scale squared and a standard
 * deviation carries it once.
 *
 * `total` is the published total and may carry a value a stage reported at completion.
 * `sampleTotal`, `count` and `welfordSumSqDev` describe the task updates alone, so an out of
 * order or duplicate stage event cannot shift the mean the deviations were measured against.
 *
 * `min` and `max` read 0 on a record built from a stage completion event with no task update
 * behind it, so at `count == 0` they are placeholders rather than observed values. Read them
 * through `sampleMin` and `sampleMax` unless the caller wants that placeholder.
 */
case class StatisticsMetrics(min: Long, med: Long, max: Long, count: Long, total: Long,
    welfordSumSqDev: Double = 0.0, sampleTotal: Long = 0L) {

  /** Sample standard deviation in stored units, or None with fewer than two samples. */
  def stddev: Option[Double] = StatisticsMetrics.sampleStddev(welfordSumSqDev, count)

  /**
   * Smallest and largest task update, or None when no task reported. The GPU metric reports
   * read the extrema through these; the plan and Photon paths still read the raw fields.
   */
  def sampleMin: Option[Long] = if (count > 0L) Some(min) else None
  def sampleMax: Option[Long] = if (count > 0L) Some(max) else None
}

object StatisticsMetrics {
  /**
   * Sample standard deviation from an accumulated deviation sum, in whatever units that sum was
   * accumulated in. None below two samples, where the n-1 denominator is zero and the result
   * would be NaN. Every caller shares this guard so the boundary cannot drift between them.
   */
  def sampleStddev(welfordSumSqDev: Double, count: Long): Option[Double] = {
    if (count < 2L) None else Some(Math.sqrt(Math.max(0.0, welfordSumSqDev) / (count - 1L)))
  }

  /**
   * Chan's parallel form: combines two accumulated deviation sums. The correction term carries
   * the squared gap between the two means, which a plain addition would drop, so merging two
   * groups that sat at different levels keeps the spread between them.
   */
  def mergeSumSqDev(aCount: Long, aTotal: Long, aDev: Double,
      bCount: Long, bTotal: Long, bDev: Double): Double = {
    if (aCount == 0L || bCount == 0L) {
      aDev + bDev
    } else {
      val delta = bTotal.toDouble / bCount - aTotal.toDouble / aCount
      aDev + bDev + delta * delta * aCount * bCount / (aCount + bCount)
    }
  }

  // a static variable used to represent zero-statistics instead of allocating a dummy record
  // on every calculation.
  val ZERO_RECORD: StatisticsMetrics = StatisticsMetrics(0L, 0L, 0L, 0L, 0L, 0.0, 0L)

  def createFromArr(arr: Array[Long]): StatisticsMetrics = {
    if (arr.isEmpty) {
      return ZERO_RECORD
    }
    val medV = findMedianInPlace(arr)(chooseMidpointPivotInPlace)
    var minV = Long.MaxValue
    var maxV = Long.MinValue
    var totalV = 0L
    var meanV = 0.0
    var sumSqDevV = 0.0
    var seen = 0L
    arr.foreach { v =>
      if (v < minV) {
        minV = v
      }
      if (v > maxV) {
        maxV = v
      }
      totalV += v
      seen += 1L
      val delta = v.toDouble - meanV
      meanV += delta / seen
      sumSqDevV += delta * (v.toDouble - meanV)
    }
    StatisticsMetrics(minV, medV, maxV, arr.length, totalV, sumSqDevV, totalV)
  }

  def createOptionalFromArr(arr: Array[Long]): Option[StatisticsMetrics] = {
    if (arr.isEmpty) {
      return None
    }
    Some(createFromArr(arr))
  }
}
