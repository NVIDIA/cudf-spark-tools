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

package com.nvidia.spark.rapids.tool.tuning

import com.nvidia.spark.rapids.tool.profiling.StageAggGpuMetricsProfileResult
import com.nvidia.spark.rapids.tool.tuning.RuntimeConfigParser.ParsedBytes
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

class GpuConcurrencySeedSuite extends AnyFunSuite {
  import GpuConcurrencySeed._

  private val MiB = 1024L * 1024L
  private val GiB = 1024L * MiB

  private def metric(
      stageId: Int,
      metricName: String = FootprintMetricName,
      max: Option[Long],
      count: Long = 4L,
      sampleTotal: Option[Long] = Some(200L),
      welfordSumSqDev: Double = 25.0): StageAggGpuMetricsProfileResult = {
    StageAggGpuMetricsProfileResult(
      stageId = stageId,
      numTasks = 8,
      metricName = metricName,
      unit = "bytes",
      total = None,
      max = max,
      count = count,
      min = Some(10L),
      welfordSumSqDev = welfordSumSqDev,
      sampleTotal = sampleTotal)
  }

  private def rightValue[A](result: Either[String, A]): A = result match {
    case Right(value) => value
    case Left(error) => fail(error)
  }

  test("compareBatches distinguishes equal bytes, changed bytes, and unknown values") {
    compareBatches(Some("2g"), Some(TargetBatch("2048m", PreservedBatch))) shouldBe
      CompatibleBatch(ParsedBytes(2L * GiB), ParsedBytes(2L * GiB), PreservedBatch)

    compareBatches(Some("2g"), Some(TargetBatch("2147483647", RecommendedBatch))) shouldBe
      ChangedBatch(ParsedBytes(2L * GiB), ParsedBytes(2147483647L), RecommendedBatch)

    compareBatches(None, Some(TargetBatch("2g", EnforcedBatch))) shouldBe UnknownBatch
    compareBatches(Some("2g"), Some(TargetBatch("1GiB", LimitedBatch))) shouldBe UnknownBatch
  }

  test("selectFootprint uses the largest positive maximum and retains diagnostics") {
    val selectedRow = metric(
      stageId = 7,
      max = Some(900L),
      count = 3L,
      sampleTotal = Some(1500L),
      welfordSumSqDev = 180000.0)
    val rows = Seq(
      metric(stageId = 1, metricName = "gpuSpill", max = Some(5000L)),
      metric(stageId = 2, max = None),
      metric(stageId = 3, max = Some(0L)),
      metric(stageId = 4, max = Some(-1L)),
      metric(stageId = 5, max = Some(800L)),
      selectedRow)

    selectFootprint(rows) shouldBe Some(FootprintEvidence(
      stageId = selectedRow.stageId,
      maxBytes = selectedRow.max.get,
      sampleCount = selectedRow.count,
      cv = selectedRow.cv,
      maxOverMean = selectedRow.maxOverMean))
  }

  test("selectFootprint returns empty without a positive footprint row") {
    selectFootprint(Seq(
      metric(stageId = 1, metricName = "gpuSpill", max = Some(100L)),
      metric(stageId = 2, max = None),
      metric(stageId = 3, max = Some(0L)))) shouldBe None
  }

  test("selectFootprint accepts one sample and keeps the first maximum on ties") {
    val firstMaximum = metric(
      stageId = 7,
      max = Some(900L),
      count = 1L,
      sampleTotal = Some(900L),
      welfordSumSqDev = 0.0)
    val tiedMaximum = metric(
      stageId = 9,
      max = Some(900L),
      count = 3L,
      sampleTotal = Some(1800L),
      welfordSumSqDev = 100.0)

    selectFootprint(Seq(firstMaximum, tiedMaximum)) shouldBe Some(FootprintEvidence(
      stageId = firstMaximum.stageId,
      maxBytes = firstMaximum.max.get,
      sampleCount = 1L,
      cv = None,
      maxOverMean = Some(1.0)))
  }

  test("selectFootprint retains aggregate updates beyond the stage task count") {
    val retriedAttempts = metric(
      stageId = 11,
      max = Some(900L),
      count = 10L)

    val selected = selectFootprint(Seq(
      metric(stageId = 10, max = Some(800L), count = 8L),
      retriedAttempts)).get

    selected.stageId shouldBe 11
    selected.maxBytes shouldBe 900L
    selected.sampleCount shouldBe 10L
  }

  test("calculateRecommendation uses the unpadded quotient without a default max-four cap") {
    val result = rightValue(calculateRecommendation(
      rmmPoolBytes = 1024L * MiB,
      footprintBytes = 100L * MiB,
      taskSlotsPerGpu = 20L,
      configuredCountCap = None,
      explicitToolCap = None))

    result.concurrentGpuTasks shouldBe 10L
    result.rmmPoolToFootprintCandidate shouldBe 10L
    result.taskSlotCap shouldBe 20L
    result.permitRepresentabilityCap shouldBe 32L
    result.footprintExceedsRmmPool shouldBe false
  }

  test("calculateRecommendation applies every runtime and operator cap") {
    rightValue(calculateRecommendation(
      1024L * MiB, 100L * MiB, 6L, None, None)).concurrentGpuTasks shouldBe 6L

    rightValue(calculateRecommendation(
      64L * MiB, 1L * MiB, 100L, None, None)).concurrentGpuTasks shouldBe 2L

    rightValue(calculateRecommendation(
      1024L * MiB, 100L * MiB, 20L, Some(7L), None)).concurrentGpuTasks shouldBe 7L

    rightValue(calculateRecommendation(
      1024L * MiB, 100L * MiB, 20L, None, Some(5L))).concurrentGpuTasks shouldBe 5L

    val combined = rightValue(calculateRecommendation(
      1024L * MiB, 100L * MiB, 20L, Some(7L), Some(5L)))
    combined.concurrentGpuTasks shouldBe 5L
    combined.configuredCountCap shouldBe Some(7L)
    combined.explicitToolCap shouldBe Some(5L)
  }

  test("calculateRecommendation emits one when the footprint exceeds the pool") {
    val result = rightValue(calculateRecommendation(
      rmmPoolBytes = 64L * MiB,
      footprintBytes = 65L * MiB,
      taskSlotsPerGpu = 8L,
      configuredCountCap = None,
      explicitToolCap = None))

    result.concurrentGpuTasks shouldBe 1L
    result.rmmPoolToFootprintCandidate shouldBe 1L
    result.footprintExceedsRmmPool shouldBe true
  }

  test("calculateRecommendation rejects non-positive required inputs") {
    calculateRecommendation(0L, 1L, 1L, None, None).isLeft shouldBe true
    calculateRecommendation(1L, 0L, 1L, None, None).isLeft shouldBe true
    calculateRecommendation(1L, 1L, 0L, None, None).isLeft shouldBe true
  }

}
