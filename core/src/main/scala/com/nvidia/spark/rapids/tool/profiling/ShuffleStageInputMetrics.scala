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

package com.nvidia.spark.rapids.tool.profiling

/**
 * Describes how the shuffle input bytes of a consumer stage were obtained.
 *
 * Profiling reads measured `GpuColumnarExchange` data sizes, while qualification reads regular
 * CPU `Exchange` data sizes and must estimate the GPU input from them.
 */
sealed abstract class ShuffleInputProvenance(val label: String)

object ShuffleInputProvenance {
  /** GPU event logs expose the exchange data size that the GPU actually processed. */
  case object Measured extends ShuffleInputProvenance("measured")
  /** CPU event logs expose CPU exchange data size, which only estimates the GPU input. */
  case object Estimated extends ShuffleInputProvenance("estimated")
}

/**
 * Total uncompressed shuffle input entering a single consumer-stage execution, together with the
 * safety evidence of the stage attempt that was selected to represent it.
 *
 * One record is produced per (SQL execution, consumer stage). `totalShuffleInputBytes` sums every
 * distinct shuffle branch that feeds the stage, so an exchange reused through two branches into the
 * same stage contributes twice.
 *
 * The record is intentionally raw: it carries measured bytes only. The tool-specific input-size
 * factor belongs to AutoTuner configuration and is applied by the downward policy, not here.
 *
 * @param sqlId              SQL execution the consumer stage belongs to
 * @param stageId            consumer stage id
 * @param stageAttemptId     attempt selected as the representative successful attempt
 * @param totalShuffleInputBytes sum of the uncompressed `data size` of every incoming branch
 * @param numShuffleBranches number of distinct incoming shuffle branches that were summed
 * @param numTasks           declared task count of the selected attempt
 * @param hasPositiveSpill   true when any successful task of the selected attempt spilled to
 *                           host/memory or disk, or when GPU SQL spill was recorded for it
 * @param hasSkew            true when the selected attempt shows shuffle read skew
 */
case class ShuffleStageInputRecord(
    sqlId: Long,
    stageId: Int,
    stageAttemptId: Int,
    totalShuffleInputBytes: Long,
    numShuffleBranches: Int,
    numTasks: Int,
    hasPositiveSpill: Boolean,
    hasSkew: Boolean)

/**
 * Reason the shuffle-stage input inventory could not be proven complete.
 *
 * A global reduction is more dangerous than an overestimate, so any single reason makes the whole
 * analysis unusable rather than only dropping the affected stage.
 */
sealed abstract class ShuffleStageInputIncompleteReason(val description: String)

object ShuffleStageInputIncompleteReason {
  /** An executed non-broadcast shuffle input had no supported `data size` metric. */
  case class MissingExchangeMetric(sqlId: Long, nodeId: Long, nodeName: String)
    extends ShuffleStageInputIncompleteReason(
      s"SQL $sqlId node $nodeId ($nodeName) has no supported shuffle 'data size' metric")

  /** A shuffle node type that this analysis cannot size was executed. */
  case class UnsupportedShuffleNode(sqlId: Long, nodeId: Long, nodeName: String)
    extends ShuffleStageInputIncompleteReason(
      s"SQL $sqlId node $nodeId ($nodeName) is an unsupported shuffle input")

  /** The downstream consumer stage of an exchange branch could not be resolved unambiguously. */
  case class UnresolvedConsumerStage(sqlId: Long, nodeId: Long, nodeName: String)
    extends ShuffleStageInputIncompleteReason(
      s"SQL $sqlId node $nodeId ($nodeName) has no reliable consumer-stage mapping")

  /** The SQL execution never reached a terminal successful end event. */
  case class IncompleteSqlExecution(sqlId: Long)
    extends ShuffleStageInputIncompleteReason(
      s"SQL $sqlId did not complete successfully")

  /** No completed, non-failed attempt exists for a consumer stage. */
  case class NoSuccessfulStageAttempt(sqlId: Long, stageId: Int)
    extends ShuffleStageInputIncompleteReason(
      s"SQL $sqlId consumer stage $stageId has no completed successful attempt")
}

/**
 * Raw result of the consumer-stage shuffle input analysis for one application.
 *
 * An application that genuinely executed no shuffle exchange is still `analyzed`: it produces no
 * records and no incomplete reasons, and the downward pass simply has nothing to size. That is a
 * different state from a provider that never produced an analysis at all, which must fail closed.
 *
 * @param records            one record per (SQL execution, consumer stage)
 * @param incompleteReasons  non-empty when the inventory is not trustworthy; the downward pass
 *                           must then keep the normal recommendation
 * @param provenance         whether `records` carry measured or estimated shuffle bytes
 * @param analyzed           false when no analysis was produced for the application at all
 * @param appHasFailedStage  true when any stage in the application failed, anywhere. This is an
 *                           application-wide signal, deliberately broader than the per-record
 *                           attempt evidence: a run that failed a stage is not a run to size a
 *                           global reduction from.
 */
case class ShuffleStageInputAnalysis(
    records: Seq[ShuffleStageInputRecord],
    incompleteReasons: Seq[ShuffleStageInputIncompleteReason],
    provenance: ShuffleInputProvenance,
    analyzed: Boolean = true,
    appHasFailedStage: Boolean = false) {

  def isComplete: Boolean = analyzed && incompleteReasons.isEmpty

  /** First few reasons, used to keep the user-facing diagnostic concise. */
  def incompleteSummary(maxReasons: Int = 2): String = {
    if (!analyzed) {
      ShuffleStageInputAnalysis.notAnalyzedSummary
    } else {
      val shown = incompleteReasons.take(maxReasons).map(_.description)
      val suffix = if (incompleteReasons.size > shown.size) {
        s" (and ${incompleteReasons.size - shown.size} more)"
      } else {
        ""
      }
      shown.mkString("; ") + suffix
    }
  }
}

object ShuffleStageInputAnalysis {
  val notAnalyzedSummary = "no shuffle input analysis was produced for the application"

  /**
   * An analysis that carries no evidence at all, used by providers that cannot produce one.
   * It always fails the completeness gate.
   */
  def empty(provenance: ShuffleInputProvenance): ShuffleStageInputAnalysis = {
    ShuffleStageInputAnalysis(Seq.empty, Seq.empty, provenance, analyzed = false)
  }
}
