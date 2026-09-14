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
import com.nvidia.spark.rapids.tool.tuning.RuntimeConfigParser.{parsePositiveBytes,
  ParsedBytes}

/** Pure issue-specific policy used to seed concurrent GPU tasks from profiling evidence. */
private[tuning] object GpuConcurrencySeed {
  val FootprintMetricName = "gpuMaxTaskFootprint"

  sealed trait TargetBatchOrigin {
    def label: String
  }

  case object RecommendedBatch extends TargetBatchOrigin {
    override val label: String = "recommended"
  }

  case object EnforcedBatch extends TargetBatchOrigin {
    override val label: String = "enforced"
  }

  case object PreservedBatch extends TargetBatchOrigin {
    override val label: String = "preserved"
  }

  case object LimitedBatch extends TargetBatchOrigin {
    override val label: String = "caller-limited"
  }

  case class TargetBatch(rawValue: String, origin: TargetBatchOrigin)

  sealed trait BatchCompatibility

  case class CompatibleBatch(source: ParsedBytes, target: ParsedBytes, origin: TargetBatchOrigin)
    extends BatchCompatibility

  case class ChangedBatch(source: ParsedBytes, target: ParsedBytes, origin: TargetBatchOrigin)
    extends BatchCompatibility

  case object UnknownBatch extends BatchCompatibility

  case class FootprintEvidence(
      stageId: Int,
      maxBytes: Long,
      sampleCount: Long,
      cv: Option[Double],
      maxOverMean: Option[Double])

  /**
   * Final concurrent-GPU-task seed and the independently calculated candidates or caps that
   * produced it. All fields are derived by [[calculateRecommendation]].
   */
  case class Recommendation(
      concurrentGpuTasks: Long,
      rmmPoolToFootprintCandidate: Long,
      taskSlotCap: Long,
      permitRepresentabilityCap: Long,
      configuredCountCap: Option[Long],
      explicitToolCap: Option[Long],
      footprintExceedsRmmPool: Boolean)

  def compareBatches(
      sourceValue: Option[String],
      targetValue: Option[TargetBatch]): BatchCompatibility = {
    (sourceValue.flatMap(parsePositiveBytes),
        targetValue.flatMap(v => parsePositiveBytes(v.rawValue).map(_ -> v.origin))) match {
      case (Some(source), Some((target, origin))) if source.bytes == target.bytes =>
        CompatibleBatch(source, target, origin)
      case (Some(source), Some((target, origin))) =>
        ChangedBatch(source, target, origin)
      case _ => UnknownBatch
    }
  }

  /** Choose the largest positive per-stage task-footprint maximum. */
  def selectFootprint(
      rows: Seq[StageAggGpuMetricsProfileResult]): Option[FootprintEvidence] = {
    rows.iterator.filter(_.metricName == FootprintMetricName).flatMap { row =>
      row.max.filter(_ > 0L).map { maxBytes =>
        FootprintEvidence(row.stageId, maxBytes, row.count, row.cv, row.maxOverMean)
      }
    }.foldLeft(Option.empty[FootprintEvidence]) {
      case (None, candidate) => Some(candidate)
      case (Some(current), candidate) if candidate.maxBytes > current.maxBytes => Some(candidate)
      case (current, _) => current
    }
  }

  /** Apply the runtime and operator caps to the unpadded pool-to-footprint quotient. */
  def calculateRecommendation(
      rmmPoolBytes: Long,
      footprintBytes: Long,
      taskSlotsPerGpu: Long,
      configuredCountCap: Option[Long],
      explicitToolCap: Option[Long]): Either[String, Recommendation] = {
    if (footprintBytes <= 0L || taskSlotsPerGpu <= 0L) {
      Left("footprint and target task slots must be positive")
    } else {
      RapidsGpuSemaphoreModel.maxRepresentableTaskCount(rmmPoolBytes).map { permitCap =>
        val rmmPoolToFootprintCandidate = math.max(1L, rmmPoolBytes / footprintBytes)
        val positiveConfiguredCap = configuredCountCap.filter(_ > 0L)
        val positiveToolCap = explicitToolCap.filter(_ > 0L)
        val concurrentGpuTasks =
          (Seq(rmmPoolToFootprintCandidate, taskSlotsPerGpu, permitCap) ++
            positiveConfiguredCap.toSeq ++ positiveToolCap.toSeq).min
        Recommendation(
          concurrentGpuTasks = concurrentGpuTasks,
          rmmPoolToFootprintCandidate = rmmPoolToFootprintCandidate,
          taskSlotCap = taskSlotsPerGpu,
          permitRepresentabilityCap = permitCap,
          configuredCountCap = positiveConfiguredCap,
          explicitToolCap = positiveToolCap,
          footprintExceedsRmmPool = footprintBytes > rmmPoolBytes)
      }
    }
  }

}
