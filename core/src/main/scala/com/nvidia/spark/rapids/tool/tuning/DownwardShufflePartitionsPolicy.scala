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

import scala.util.Try

import com.nvidia.spark.rapids.tool.profiling.{ShuffleInputProvenance, ShuffleStageInputAnalysis, ShuffleStageInputRecord}
import com.nvidia.spark.rapids.tool.tuning.config.TuningConfigProvider

import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.util.StringUtils

/**
 * Validated policy inputs of the downward-only shuffle partition pass.
 *
 * Every field is user-overridable through the tuning-config mechanism. The values are validated as
 * one unit before any recommendation can be mutated, so a partially valid configuration can never
 * produce a partially applied reduction.
 *
 * @param enabled                  master switch for the downward pass
 * @param targetPartitionSizeBytes estimated GPU input a single partition should process
 * @param inputSizeFactor          factor converting measured shuffle bytes to estimated GPU bytes
 */
case class DownwardShufflePolicyConfig(
    enabled: Boolean,
    targetPartitionSizeBytes: Long,
    inputSizeFactor: Double)

object DownwardShufflePolicyConfig {
  val ENABLED_KEY = "DOWNWARD_SHUFFLE_ENABLED"
  val TARGET_PARTITION_SIZE_KEY = "DOWNWARD_SHUFFLE_TARGET_PARTITION_SIZE"
  val INPUT_SIZE_FACTOR_KEY = "DOWNWARD_SHUFFLE_INPUT_SIZE_FACTOR"

  /** Config used when the feature is switched off. The remaining fields are never read. */
  val disabled: DownwardShufflePolicyConfig = DownwardShufflePolicyConfig(
    enabled = false,
    targetPartitionSizeBytes = 0L,
    inputSizeFactor = 0.0)

  /**
   * Reads and validates every policy entry from the tuning-config provider.
   *
   * All problems are collected so that an invalid configuration surfaces as one fail-closed
   * decision. When the feature is explicitly disabled the remaining entries are not validated,
   * which keeps a disabled run quiet even if its unused policy entries are malformed.
   *
   * @return the validated config, or the list of validation errors
   */
  def fromProvider(
      configProvider: TuningConfigProvider): Either[Seq[String], DownwardShufflePolicyConfig] = {
    rawValue(configProvider, ENABLED_KEY).flatMap(parseStrictBoolean(ENABLED_KEY, _)) match {
      case Left(err) => Left(Seq(err))
      case Right(false) => Right(disabled)
      case Right(true) => parseEnabledConfig(configProvider)
    }
  }

  private def parseEnabledConfig(
      configProvider: TuningConfigProvider): Either[Seq[String], DownwardShufflePolicyConfig] = {
    val targetSize = parseMemoryBytes(configProvider, TARGET_PARTITION_SIZE_KEY)
    val factor = parseDouble(configProvider, INPUT_SIZE_FACTOR_KEY, min = 0.0, minInclusive = false)

    val errors = Seq(targetSize, factor).collect {
      case Left(err) => err
    }
    (targetSize, factor) match {
      case (Right(size), Right(f)) if errors.isEmpty =>
        Right(DownwardShufflePolicyConfig(
          enabled = true,
          targetPartitionSizeBytes = size,
          inputSizeFactor = f))
      case _ => Left(errors)
    }
  }

  private def rawValue(
      configProvider: TuningConfigProvider, key: String): Either[String, String] = {
    Try(configProvider.getEntry(key).getDefault).toOption
      .filter(v => v != null && v.trim.nonEmpty)
      .toRight(s"'$key' is not defined")
  }

  /** Accepts only an exact 'true' or 'false'; anything else fails closed. */
  private def parseStrictBoolean(key: String, value: String): Either[String, Boolean] = {
    value.trim.toLowerCase match {
      case "true" => Right(true)
      case "false" => Right(false)
      case other => Left(s"'$key' must be exactly 'true' or 'false' but was '$other'")
    }
  }

  private def parseMemoryBytes(
      configProvider: TuningConfigProvider, key: String): Either[String, Long] = {
    rawValue(configProvider, key).flatMap { raw =>
      Try(StringUtils.convertMemorySizeToBytes(raw, Some(ByteUnit.BYTE))).toOption
        .filter(_ > 0L)
        .toRight(s"'$key' must be a positive memory size but was '$raw'")
    }
  }

  private def parseDouble(
      configProvider: TuningConfigProvider,
      key: String,
      min: Double,
      minInclusive: Boolean): Either[String, Double] = {
    val bound = if (minInclusive) s">= $min" else s"> $min"
    rawValue(configProvider, key).flatMap { raw =>
      Try(raw.trim.toDouble).toOption
        .filter(v => !v.isNaN && !v.isInfinite && (if (minInclusive) v >= min else v > min))
        .toRight(s"'$key' must be a finite number $bound but was '$raw'")
    }
  }
}

/** Reason the downward pass left the normal recommendation unchanged. */
sealed abstract class DownwardShuffleSkipReason(val description: String) {
  /**
   * Warning-level reasons point at something the user may want to fix, so they earn one concise
   * comment. Ordinary policy and safety decisions stay log-only to avoid broad output churn.
   */
  def isWarning: Boolean = false
}

object DownwardShuffleSkipReason {
  case object Disabled
    extends DownwardShuffleSkipReason("the downward shuffle partition pass is disabled")

  /**
   * The analysis ran but found a gap it cannot reason about. This is actionable, so it earns one
   * concise comment in addition to the log.
   */
  case class IncompleteEvidence(summary: String)
    extends DownwardShuffleSkipReason(
      s"shuffle input evidence is incomplete: $summary") {
    override def isWarning: Boolean = true
  }

  /**
   * No analysis was produced for the application at all, for example because the caller does not
   * wire one up. There is nothing for the user to act on, so this stays log-only.
   */
  case object NoAnalysisAvailable
    extends DownwardShuffleSkipReason(
      "no shuffle input analysis is available for this application")

  case object NoStageEvidence
    extends DownwardShuffleSkipReason("no consumer stage shuffle input was found")

  /**
   * The worst stage needs more partitions than a Spark partition count can express, or covering it
   * with whole waves would. Failing closed here is safer than recommending a partition count below
   * the requirement.
   */
  case class RequirementOutOfRange(requirement: Long)
    extends DownwardShuffleSkipReason(
      s"the worst consumer stage requires $requirement partitions, which cannot be rounded up to" +
        s" whole cluster waves within the largest representable partition count" +
        s" (${Int.MaxValue})") {
    override def isWarning: Boolean = true
  }

  /**
   * The recommended cluster shape is not known, so there is no wave to quantize to. This is the
   * common qualification case rather than something the user can act on, so it stays log-only.
   */
  case object SlotCountUnavailable
    extends DownwardShuffleSkipReason(
      "the recommended cluster's task slot count could not be determined")

  case class NotDownward(candidate: Int, normalValue: Int)
    extends DownwardShuffleSkipReason(
      s"candidate $candidate does not lower the current recommendation $normalValue")

  /**
   * The application failed a stage or hit an OOM anywhere. Sizing a global reduction from a run
   * that did not complete cleanly is unsafe even when the consumer stages themselves look healthy.
   */
  case object ApplicationHasFailedStage
    extends DownwardShuffleSkipReason("the application had a failed stage")

  case object ApplicationHadOom
    extends DownwardShuffleSkipReason("the application had an out-of-memory failure")

  case class UpwardSafetyReason(reason: String)
    extends DownwardShuffleSkipReason(
      s"an existing upward recommendation must be preserved: $reason")

  case class StagePressure(stageId: Int, reason: String)
    extends DownwardShuffleSkipReason(
      s"consumer stage $stageId shows $reason")

  case class PropertyNotMutable(property: String)
    extends DownwardShuffleSkipReason(
      s"partition property '$property' cannot be updated by the AutoTuner")

  case object PlatformControlledShuffle
    extends DownwardShuffleSkipReason(
      "Databricks automatic shuffle optimization is still effectively enabled")
}

/** Outcome of the pure downward policy. */
sealed trait DownwardShuffleDecision

object DownwardShuffleDecision {
  /**
   * A reduction that passed every policy gate.
   *
   * @param normalValue          the effective recommendation produced by normal tuning
   * @param selectedValue        the wave-quantized partition count to recommend instead
   * @param determiningRecord    the consumer stage that produced the worst requirement
   * @param estimatedInputBytes  determining stage bytes after the input-size factor
   * @param rawRequirement       partition count before rounding up to whole waves
   * @param slotCount            task slots of the recommended cluster; one execution wave
   * @param waveCount            whole waves the selected value spans
   * @param provenance           whether the bytes were measured or estimated
   * @param inputSizeFactor      factor that was applied
   */
  case class Applied(
      normalValue: Int,
      selectedValue: Int,
      determiningRecord: ShuffleStageInputRecord,
      estimatedInputBytes: Long,
      rawRequirement: Long,
      slotCount: Int,
      waveCount: Int,
      provenance: ShuffleInputProvenance,
      inputSizeFactor: Double) extends DownwardShuffleDecision

  case class Skipped(reason: DownwardShuffleSkipReason) extends DownwardShuffleDecision

  case class InvalidConfig(errors: Seq[String]) extends DownwardShuffleDecision
}

/**
 * Pure, deterministic calculator for the downward-only shuffle partition pass.
 *
 * This object never reads or mutates AutoTuner state. It converts raw consumer-stage shuffle input
 * records into either an applied reduction or a typed no-op reason, using overflow-safe arithmetic
 * and quantizing the result to whole execution waves of the recommended cluster.
 */
object DownwardShufflePartitionsPolicy {

  /** A consumer stage together with its derived estimated bytes and partition requirement. */
  private case class StageRequirement(
      record: ShuffleStageInputRecord,
      estimatedBytes: Long,
      requirement: Long)

  /**
   * Evaluates the policy for one application.
   *
   * @param configResult    validated policy config, or its validation errors
   * @param normalValue     effective shuffle partition recommendation after normal tuning
   * @param slotCount       task slots of the recommended cluster, or None when it is unknown
   * @param analysis        raw consumer-stage shuffle input analysis
   */
  def decide(
      configResult: Either[Seq[String], DownwardShufflePolicyConfig],
      normalValue: Int,
      slotCount: Option[Int],
      analysis: ShuffleStageInputAnalysis): DownwardShuffleDecision = {
    configResult match {
      case Left(errors) => DownwardShuffleDecision.InvalidConfig(errors)
      case Right(config) if !config.enabled =>
        DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.Disabled)
      case Right(config) => decideWithConfig(config, normalValue, slotCount, analysis)
    }
  }

  private def decideWithConfig(
      config: DownwardShufflePolicyConfig,
      normalValue: Int,
      slotCount: Option[Int],
      analysis: ShuffleStageInputAnalysis): DownwardShuffleDecision = {
    if (!analysis.analyzed) {
      return DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NoAnalysisAvailable)
    }
    if (!analysis.isComplete) {
      return DownwardShuffleDecision.Skipped(
        DownwardShuffleSkipReason.IncompleteEvidence(analysis.incompleteSummary()))
    }
    if (analysis.records.isEmpty) {
      return DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NoStageEvidence)
    }

    // A slot count is what a wave is measured in, so without one there is nothing to quantize to.
    val slots = slotCount.filter(_ > 0).getOrElse {
      return DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.SlotCountUnavailable)
    }

    val worst = selectWorstStage(config, analysis.records)
    waveQuantized(worst.requirement, slots) match {
      case None =>
        // No representable whole-wave count covers the requirement: fail closed rather than
        // recommend a partition count that is known to be too small.
        DownwardShuffleDecision.Skipped(
          DownwardShuffleSkipReason.RequirementOutOfRange(worst.requirement))
      case Some(candidate) if candidate >= normalValue =>
        // Also covers the case where the normal value is already at or below one wave, because
        // every candidate is at least the slot count.
        DownwardShuffleDecision.Skipped(
          DownwardShuffleSkipReason.NotDownward(candidate, normalValue))
      case Some(candidate) =>
        DownwardShuffleDecision.Applied(
          normalValue = normalValue,
          selectedValue = candidate,
          determiningRecord = worst.record,
          estimatedInputBytes = worst.estimatedBytes,
          rawRequirement = worst.requirement,
          slotCount = slots,
          waveCount = candidate / slots,
          provenance = analysis.provenance,
          inputSizeFactor = config.inputSizeFactor)
    }
  }

  /**
   * Picks the consumer stage with the largest partition requirement.
   *
   * Ties are broken by the larger estimated input, then by ascending SQL id, stage id, and attempt
   * id, so the determining stage is reproducible across runs regardless of record ordering.
   */
  private def selectWorstStage(
      config: DownwardShufflePolicyConfig,
      records: Seq[ShuffleStageInputRecord]): StageRequirement = {
    records.map { record =>
      val estimated = estimateGpuInputBytes(record.totalShuffleInputBytes, config.inputSizeFactor)
      StageRequirement(record, estimated, partitionRequirement(estimated,
        config.targetPartitionSizeBytes))
    }.reduceLeft { (best, candidate) =>
      if (isWorseThan(candidate, best)) candidate else best
    }
  }

  /** Strict "is a worse (larger) requirement than" comparison implementing the tie-break order. */
  private def isWorseThan(left: StageRequirement, right: StageRequirement): Boolean = {
    if (left.requirement != right.requirement) {
      left.requirement > right.requirement
    } else if (left.estimatedBytes != right.estimatedBytes) {
      left.estimatedBytes > right.estimatedBytes
    } else if (left.record.sqlId != right.record.sqlId) {
      left.record.sqlId < right.record.sqlId
    } else if (left.record.stageId != right.record.stageId) {
      left.record.stageId < right.record.stageId
    } else {
      left.record.stageAttemptId < right.record.stageAttemptId
    }
  }

  /**
   * Applies the tool-specific input-size factor. The product is computed in `Double` and clamped so
   * that a huge stage total can never wrap around to a small positive `Long`.
   */
  private[tuning] def estimateGpuInputBytes(totalBytes: Long, factor: Double): Long = {
    if (totalBytes <= 0L) {
      0L
    } else {
      val scaled = math.ceil(totalBytes.toDouble * factor)
      if (scaled >= Long.MaxValue.toDouble) Long.MaxValue else scaled.toLong
    }
  }

  /**
   * Ceiling of estimated bytes divided by the target partition size, with a lower bound of 1 so a
   * stage that carries any shuffle input always requires at least one partition.
   */
  private[tuning] def partitionRequirement(estimatedBytes: Long, targetBytes: Long): Long = {
    if (estimatedBytes <= 0L) {
      1L
    } else {
      // targetBytes is validated to be positive, so this cannot divide by zero and neither
      // branch can overflow because estimatedBytes is at most Long.MaxValue.
      val quotient = estimatedBytes / targetBytes
      if (estimatedBytes % targetBytes == 0L) math.max(quotient, 1L) else quotient + 1L
    }
  }

  /**
   * Rounds a raw requirement up to a whole number of execution waves of the recommended cluster.
   *
   * The slot count is a hard floor as well as the quantum, so a stage needing fewer partitions than
   * the cluster has slots still gets exactly one full wave. Every arithmetic step stays in `Long`
   * and the result is range-checked, because `ceil(raw / slots) * slots` can exceed the largest
   * partition count Spark can express even when the requirement itself does not.
   *
   * @return the wave-quantized candidate, or None when no whole-wave count within `Int.MaxValue`
   *         covers the requirement
   */
  private[tuning] def waveQuantized(requirement: Long, slots: Int): Option[Int] = {
    if (slots <= 0 || requirement > Int.MaxValue.toLong) {
      return None
    }
    // Both operands are at most Int.MaxValue here, so neither the sum nor the product overflows.
    val raw = math.max(slots.toLong, requirement)
    val candidate = ((raw + slots - 1L) / slots) * slots
    if (candidate > Int.MaxValue.toLong) None else Some(candidate.toInt)
  }

  /**
   * Builds the single user-facing comment for an applied reduction. It names every input of the
   * decision so the recommendation can be audited without re-running the tool.
   */
  def appliedComment(
      partitionUpdates: Seq[(String, Int)],
      decision: DownwardShuffleDecision.Applied): String = {
    val record = decision.determiningRecord
    // Normally every property lands on the same value, and the comment reads as one sentence about
    // both. They can only differ if a property was already below the candidate and got clamped to
    // its own value, in which case each is named with what it actually became.
    val appliedValues = partitionUpdates.map(_._2).distinct
    val lowered = if (appliedValues.size == 1) {
      s"${partitionUpdates.map(p => s"'${p._1}'").mkString(" and ")} lowered from " +
        s"${decision.normalValue} to ${appliedValues.head}"
    } else {
      partitionUpdates.map { case (property, value) => s"'$property' lowered to $value" }
        .mkString(" and ") + s", from an effective ${decision.normalValue}"
    }
    lowered + s" based on the " +
      s"${decision.provenance.label} shuffle input of the worst consumer stage " +
      s"(SQL ${record.sqlId}, stage ${record.stageId}, attempt ${record.stageAttemptId}): " +
      s"${record.totalShuffleInputBytes} bytes across ${record.numShuffleBranches} shuffle " +
      s"branch(es), input size factor ${decision.inputSizeFactor}, " +
      s"${decision.estimatedInputBytes} estimated bytes, " +
      s"raw requirement ${decision.rawRequirement} partitions rounded up to " +
      s"${decision.waveCount} execution wave(s) of ${decision.slotCount} cluster task slots " +
      s"(${decision.selectedValue} partitions)."
  }
}
