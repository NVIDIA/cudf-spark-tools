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

import com.nvidia.spark.rapids.tool.ToolTestUtils
import com.nvidia.spark.rapids.tool.profiling.{ShuffleInputProvenance, ShuffleStageInputAnalysis, ShuffleStageInputIncompleteReason, ShuffleStageInputRecord}
import com.nvidia.spark.rapids.tool.tuning.config.{ProfTuningConfigProvider, QualTuningConfigProvider, TuningConfigEntry, TuningConfigProvider}
import org.scalatest.funsuite.AnyFunSuite

/**
 * Focused tests for the configuration and the pure downward shuffle-partition policy.
 *
 * These tests never construct an AutoTuner: the policy layer must be provable on its own.
 */
class DownwardShufflePartitionsSuite extends AnyFunSuite {

  private val GiB = 1024L * 1024L * 1024L

  /**
   * The pass ships disabled, so these providers opt in unless the test overrides the switch
   * itself. Everything below the switch is then exercised against the shipped defaults.
   */
  private def withOptIn(default: List[TuningConfigEntry]): List[TuningConfigEntry] = {
    if (default.exists(_.name == DownwardShufflePolicyConfig.ENABLED_KEY)) {
      default
    } else {
      TuningConfigEntry(name = DownwardShufflePolicyConfig.ENABLED_KEY, default = "true") :: default
    }
  }

  private def profProvider(
      default: List[TuningConfigEntry] = List.empty): ProfTuningConfigProvider = {
    TuningConfigProvider.builder
      .withUserProvidedConfig(Some(ToolTestUtils.buildTuningConfigs(default = withOptIn(default))))
      .build[ProfTuningConfigProvider]
  }

  private def qualProvider(
      default: List[TuningConfigEntry] = List.empty,
      qualification: List[TuningConfigEntry] = List.empty): QualTuningConfigProvider = {
    TuningConfigProvider.builder
      .withUserProvidedConfig(Some(ToolTestUtils.buildTuningConfigs(
        default = withOptIn(default), qualification = qualification)))
      .build[QualTuningConfigProvider]
  }

  /** Provider with no user overrides at all, so it reflects exactly what the tool ships. */
  private def shippedProfProvider(): ProfTuningConfigProvider = {
    TuningConfigProvider.builder.build[ProfTuningConfigProvider]
  }

  private def record(
      sqlId: Long = 0L,
      stageId: Int = 1,
      stageAttemptId: Int = 0,
      totalBytes: Long,
      numBranches: Int = 1,
      numTasks: Int = 200,
      hasSpill: Boolean = false,
      hasSkew: Boolean = false): ShuffleStageInputRecord = {
    ShuffleStageInputRecord(sqlId, stageId, stageAttemptId, totalBytes, numBranches, numTasks,
      hasSpill, hasSkew)
  }

  private def analysis(
      records: Seq[ShuffleStageInputRecord],
      provenance: ShuffleInputProvenance = ShuffleInputProvenance.Measured
  ): ShuffleStageInputAnalysis = {
    ShuffleStageInputAnalysis(records, Seq.empty, provenance)
  }

  /** Config used by the arithmetic tests: 1 GiB target, cores basis, no reduction threshold. */
  private val baseConfig = DownwardShufflePolicyConfig(
    enabled = true,
    targetPartitionSizeBytes = GiB,
    inputSizeFactor = 1.0,
    slotBasis = DownwardShuffleSlotBasis.Cores,
    minReductionFactor = 1.0)

  /** Slot count of the cluster the arithmetic tests recommend for: 125 executors x 16 cores. */
  private val slots = 2000

  /** Unwraps a config result that the test expects to be valid. */
  private def expectValid(
      result: Either[Seq[String], DownwardShufflePolicyConfig]): DownwardShufflePolicyConfig = {
    result.fold(errors => fail(s"expected a valid config but got: ${errors.mkString(", ")}"),
      identity)
  }

  /** Unwraps a config result that the test expects to be invalid. */
  private def expectErrors(
      result: Either[Seq[String], DownwardShufflePolicyConfig]): Seq[String] = {
    result.fold(identity, config => fail(s"expected validation errors but got: $config"))
  }

  private def decide(
      normalValue: Int,
      records: Seq[ShuffleStageInputRecord],
      config: DownwardShufflePolicyConfig = baseConfig,
      provenance: ShuffleInputProvenance = ShuffleInputProvenance.Measured,
      slotCount: Option[Int] = Some(slots)
  ): DownwardShuffleDecision = {
    DownwardShufflePartitionsPolicy.decide(Right(config), normalValue, slotCount,
      analysis(records, provenance))
  }

  //
  // Configuration
  //

  test("the pass ships disabled so enabling it is an explicit opt-in") {
    assert(DownwardShufflePolicyConfig.fromProvider(shippedProfProvider()) ==
      Right(DownwardShufflePolicyConfig.disabled))
  }

  test("profiling defaults match the product contract") {
    val config = expectValid(DownwardShufflePolicyConfig.fromProvider(profProvider()))
    assert(config.enabled)
    assert(config.targetPartitionSizeBytes == GiB)
    assert(config.inputSizeFactor == 1.0)
    assert(config.slotBasis == DownwardShuffleSlotBasis.Cores)
    // Wave quantization already prevents trivial reductions, so no extra threshold is imposed.
    assert(config.minReductionFactor == 1.0)
  }

  test("the retired rung entries are gone and their removal does not break config loading") {
    Seq("DOWNWARD_SHUFFLE_PARTITION_FLOOR", "DOWNWARD_SHUFFLE_RUNG_MULTIPLIER").foreach { key =>
      Seq(profProvider(), qualProvider()).foreach { provider =>
        assert(Try(provider.getEntry(key)).isFailure, s"'$key' should no longer be defined")
      }
    }
    assert(DownwardShufflePolicyConfig.fromProvider(profProvider()).isRight)
    assert(DownwardShufflePolicyConfig.fromProvider(qualProvider()).isRight)
  }

  test("qualification overrides the input size factor in its own tool section") {
    val config = expectValid(DownwardShufflePolicyConfig.fromProvider(qualProvider()))
    assert(config.enabled)
    assert(config.inputSizeFactor == 0.8)
    // Everything else still comes from the shared defaults.
    assert(config.targetPartitionSizeBytes == GiB)
    assert(config.slotBasis == DownwardShuffleSlotBasis.Cores)
  }

  test("user overrides are honored in the default and tool sections") {
    val fromDefault = DownwardShufflePolicyConfig.fromProvider(profProvider(
      default = List(
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_TARGET_PARTITION_SIZE", default = "512m"),
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_SLOT_BASIS", default = "concurrentGpuTasks"))))
    assert(fromDefault == Right(baseConfig.copy(
      targetPartitionSizeBytes = 512L * 1024L * 1024L,
      slotBasis = DownwardShuffleSlotBasis.ConcurrentGpuTasks)))

    val fromToolSection = DownwardShufflePolicyConfig.fromProvider(qualProvider(
      qualification = List(
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_INPUT_SIZE_FACTOR", default = "0.5"))))
    assert(fromToolSection == Right(baseConfig.copy(inputSizeFactor = 0.5)))
  }

  test("the slot basis accepts only a known label, case-insensitively") {
    Seq("cores" -> DownwardShuffleSlotBasis.Cores,
      "CORES" -> DownwardShuffleSlotBasis.Cores,
      " concurrentgputasks " -> DownwardShuffleSlotBasis.ConcurrentGpuTasks
    ).foreach { case (raw, expected) =>
      val result = DownwardShufflePolicyConfig.fromProvider(profProvider(
        default = List(TuningConfigEntry(name = "DOWNWARD_SHUFFLE_SLOT_BASIS", default = raw))))
      assert(result.exists(_.slotBasis == expected), s"'$raw' should parse as $expected")
    }
    Seq("gpus", "tasks", "1").foreach { raw =>
      val result = DownwardShufflePolicyConfig.fromProvider(profProvider(
        default = List(TuningConfigEntry(name = "DOWNWARD_SHUFFLE_SLOT_BASIS", default = raw))))
      assert(result.isLeft, s"'$raw' should not parse as a slot basis")
    }
  }

  test("a disabled feature short-circuits without validating the other entries") {
    val provider = profProvider(default = List(
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_ENABLED", default = "false"),
      // Deliberately invalid; it must not be read while the feature is off.
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_SLOT_BASIS", default = "not-a-basis")))
    val configResult = DownwardShufflePolicyConfig.fromProvider(provider)
    assert(configResult == Right(DownwardShufflePolicyConfig.disabled))
    assert(DownwardShufflePartitionsPolicy.decide(configResult, 4000, Some(slots),
      analysis(Seq(record(totalBytes = GiB)))) ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.Disabled))
  }

  test("the enabled switch requires an exact boolean") {
    Seq("yes", "1", "TRUE!", "0").foreach { value =>
      val result = DownwardShufflePolicyConfig.fromProvider(profProvider(
        default = List(TuningConfigEntry(name = "DOWNWARD_SHUFFLE_ENABLED", default = value))))
      assert(result.isLeft, s"'$value' should not parse as a boolean")
    }
    // Case-insensitive exact values are still accepted.
    assert(DownwardShufflePolicyConfig.fromProvider(profProvider(
      default = List(TuningConfigEntry(name = "DOWNWARD_SHUFFLE_ENABLED", default = "TRUE"))))
      .exists(_.enabled))
  }

  test("every invalid numeric boundary is rejected") {
    val invalidByKey = Seq(
      "DOWNWARD_SHUFFLE_TARGET_PARTITION_SIZE" -> Seq("0", "-1g", "abc"),
      "DOWNWARD_SHUFFLE_INPUT_SIZE_FACTOR" -> Seq("0", "-0.5", "abc"),
      // A reduction factor below 1.0 would allow raising the recommendation.
      "DOWNWARD_SHUFFLE_MIN_REDUCTION_FACTOR" -> Seq("0.9", "0", "abc"))

    invalidByKey.foreach { case (key, values) =>
      values.foreach { value =>
        val result = DownwardShufflePolicyConfig.fromProvider(profProvider(
          default = List(TuningConfigEntry(name = key, default = value))))
        assert(result.isLeft, s"'$key' = '$value' should be rejected")
        assert(result.swap.exists(_.exists(_.contains(key))),
          s"the error for '$key' = '$value' should name the key")
      }
    }
  }

  test("NaN and infinite values are rejected") {
    Seq("DOWNWARD_SHUFFLE_INPUT_SIZE_FACTOR",
      "DOWNWARD_SHUFFLE_MIN_REDUCTION_FACTOR").foreach { key =>
      Seq("NaN", "Infinity", "-Infinity").foreach { value =>
        val result = DownwardShufflePolicyConfig.fromProvider(profProvider(
          default = List(TuningConfigEntry(name = key, default = value))))
        assert(result.isLeft, s"'$key' = '$value' should be rejected")
      }
    }
  }

  test("all configuration errors are reported as one fail-closed decision") {
    val provider = profProvider(default = List(
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_SLOT_BASIS", default = "not-a-basis"),
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_MIN_REDUCTION_FACTOR", default = "0.5")))
    val configResult = DownwardShufflePolicyConfig.fromProvider(provider)
    val errors = expectErrors(configResult)
    assert(errors.size == 2)
    DownwardShufflePartitionsPolicy.decide(configResult, 4000, Some(slots),
      analysis(Seq(record(totalBytes = GiB)))) match {
      case DownwardShuffleDecision.InvalidConfig(reported) => assert(reported == errors)
      case other => fail(s"expected an invalid-config decision but got $other")
    }
  }

  //
  // Arithmetic
  //

  test("estimateGpuInputBytes rounds up and clamps instead of overflowing") {
    assert(DownwardShufflePartitionsPolicy.estimateGpuInputBytes(0L, 1.0) == 0L)
    assert(DownwardShufflePartitionsPolicy.estimateGpuInputBytes(10L, 0.8) == 8L)
    // 0.75 of 10 is 7.5 and must round up so the requirement is never understated.
    assert(DownwardShufflePartitionsPolicy.estimateGpuInputBytes(10L, 0.75) == 8L)
    assert(DownwardShufflePartitionsPolicy.estimateGpuInputBytes(Long.MaxValue, 2.0) ==
      Long.MaxValue)
  }

  test("partitionRequirement is an exact ceiling with a lower bound of one") {
    assert(DownwardShufflePartitionsPolicy.partitionRequirement(0L, GiB) == 1L)
    assert(DownwardShufflePartitionsPolicy.partitionRequirement(1L, GiB) == 1L)
    // Exact multiples of the target must not round up to an extra partition.
    assert(DownwardShufflePartitionsPolicy.partitionRequirement(1000L * GiB, GiB) == 1000L)
    assert(DownwardShufflePartitionsPolicy.partitionRequirement(1000L * GiB + 1L, GiB) == 1001L)
  }

  test("waveQuantized rounds up to whole waves and treats the slot count as a floor") {
    val cases = Seq(
      // Below one wave still gets exactly one wave.
      1L -> 2000, 1999L -> 2000,
      // Exactly one wave stays at one wave rather than rounding to two.
      2000L -> 2000,
      // One partition above a wave boundary opens the next whole wave.
      2001L -> 4000, 4001L -> 6000,
      // Several waves round up to the next whole wave, not to the requirement.
      5000L -> 6000, 6000L -> 6000)
    cases.foreach { case (requirement, expected) =>
      assert(DownwardShufflePartitionsPolicy.waveQuantized(requirement, slots) == Some(expected),
        s"requirement $requirement")
    }
  }

  test("waveQuantized handles a slot count of one and an absent-sized cluster") {
    // With one slot per wave the candidate is just the requirement, bounded below by one.
    assert(DownwardShufflePartitionsPolicy.waveQuantized(1L, 1) == Some(1))
    assert(DownwardShufflePartitionsPolicy.waveQuantized(37L, 1) == Some(37))
    // A non-positive slot count has no wave to quantize to.
    assert(DownwardShufflePartitionsPolicy.waveQuantized(1000L, 0).isEmpty)
    assert(DownwardShufflePartitionsPolicy.waveQuantized(1000L, -16).isEmpty)
  }

  test("a requirement no representable wave count can cover fails closed") {
    // Beyond the largest partition count Spark can express.
    assert(DownwardShufflePartitionsPolicy.waveQuantized(Int.MaxValue.toLong + 1L, slots).isEmpty)
    assert(DownwardShufflePartitionsPolicy.waveQuantized(Long.MaxValue, slots).isEmpty)
    // Representable on its own, but rounding it up to a whole wave overshoots Int.MaxValue.
    assert(DownwardShufflePartitionsPolicy.waveQuantized(Int.MaxValue.toLong, slots).isEmpty)
    // The exact largest representable whole-wave count is still covered.
    val largestWholeWave = (Int.MaxValue / slots) * slots
    assert(DownwardShufflePartitionsPolicy.waveQuantized(largestWholeWave.toLong, slots) ==
      Some(largestWholeWave))
  }

  test("an out-of-range requirement skips instead of recommending an uncovering value") {
    decide(Int.MaxValue, Seq(record(totalBytes = Long.MaxValue))) match {
      case DownwardShuffleDecision.Skipped(
          reason @ DownwardShuffleSkipReason.RequirementOutOfRange(_)) =>
        assert(reason.isWarning)
      case other => fail(s"expected an out-of-range skip but got $other")
    }
    // Representable requirement whose wave rounding would wrap past Int.MaxValue.
    decide(Int.MaxValue, Seq(record(totalBytes = Int.MaxValue.toLong * GiB))) match {
      case DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.RequirementOutOfRange(_)) =>
      case other => fail(s"expected an out-of-range skip but got $other")
    }
  }

  test("a missing or non-positive slot count is a quiet no-op rather than a division error") {
    Seq(None, Some(0), Some(-16)).foreach { slotCount =>
      assert(decide(8000, Seq(record(totalBytes = 900L * GiB)), slotCount = slotCount) ==
        DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.SlotCountUnavailable),
        s"slot count $slotCount")
    }
    assert(!DownwardShuffleSkipReason.SlotCountUnavailable.isWarning)
  }

  //
  // Policy decisions
  //

  test("AE1: a multi-input join stage is sized from its combined branches") {
    // Two branches whose combined input needs 730 partitions at a 1 GiB target.
    val joinStage = record(sqlId = 0L, stageId = 7, totalBytes = 729L * GiB + 1L, numBranches = 2)
    decide(normalValue = 8000, records = Seq(joinStage)) match {
      case applied: DownwardShuffleDecision.Applied =>
        assert(applied.rawRequirement == 730L)
        // Below one wave, so the cluster's own slot count is the floor.
        assert(applied.selectedValue == 2000)
        assert(applied.slotCount == 2000)
        assert(applied.waveCount == 1)
        assert(applied.normalValue == 8000)
        assert(applied.determiningRecord == joinStage)
        assert(applied.provenance == ShuffleInputProvenance.Measured)
      case other => fail(s"expected an applied reduction but got $other")
    }
  }

  test("the qualification factor lowers the requirement of the same stage") {
    val stage = record(totalBytes = 4000L * GiB)
    val qualConfig = baseConfig.copy(inputSizeFactor = 0.8)
    decide(8000, Seq(stage), qualConfig, ShuffleInputProvenance.Estimated) match {
      case applied: DownwardShuffleDecision.Applied =>
        assert(applied.estimatedInputBytes == 3200L * GiB)
        assert(applied.rawRequirement == 3200L)
        // 3200 spans two whole waves of 2000.
        assert(applied.selectedValue == 4000)
        assert(applied.waveCount == 2)
        assert(applied.inputSizeFactor == 0.8)
        assert(applied.provenance == ShuffleInputProvenance.Estimated)
      case other => fail(s"expected an applied reduction but got $other")
    }
  }

  test("the worst consumer stage determines the candidate") {
    val small = record(sqlId = 0L, stageId = 1, totalBytes = 100L * GiB)
    val worst = record(sqlId = 1L, stageId = 9, totalBytes = 2500L * GiB)
    Seq(Seq(small, worst), Seq(worst, small)).foreach { records =>
      decide(8000, records) match {
        case applied: DownwardShuffleDecision.Applied =>
          assert(applied.determiningRecord == worst)
          assert(applied.selectedValue == 4000)
        case other => fail(s"expected an applied reduction but got $other")
      }
    }
  }

  test("ties are broken by estimated bytes, then by ascending SQL and stage ids") {
    // All three round to the same requirement of 2 partitions.
    val moreBytes = record(sqlId = 5L, stageId = 3, totalBytes = GiB + 900L)
    val lowIds = record(sqlId = 1L, stageId = 2, totalBytes = GiB + 1L)
    val highIds = record(sqlId = 1L, stageId = 4, totalBytes = GiB + 1L)

    // Larger estimated bytes wins the tie even though its ids sort later.
    decide(4000, Seq(lowIds, moreBytes, highIds)) match {
      case applied: DownwardShuffleDecision.Applied =>
        assert(applied.determiningRecord == moreBytes)
      case other => fail(s"expected an applied reduction but got $other")
    }
    // With equal bytes, the lowest (sqlId, stageId) wins regardless of input order.
    Seq(Seq(lowIds, highIds), Seq(highIds, lowIds)).foreach { records =>
      decide(4000, records) match {
        case applied: DownwardShuffleDecision.Applied =>
          assert(applied.determiningRecord == lowIds)
        case other => fail(s"expected an applied reduction but got $other")
      }
    }
  }

  test("an exact wave multiple does not round up to an extra wave") {
    decide(8000, Seq(record(totalBytes = 4000L * GiB))) match {
      case applied: DownwardShuffleDecision.Applied =>
        assert(applied.rawRequirement == 4000L)
        assert(applied.selectedValue == 4000)
        assert(applied.waveCount == 2)
      case other => fail(s"expected an applied reduction but got $other")
    }
  }

  test("AE7: a candidate at or above the normal value never raises it") {
    // The candidate wave equals the current value.
    assert(decide(2000, Seq(record(totalBytes = 900L * GiB))) ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NotDownward(2000, 2000)))
    // The candidate wave is larger than the current value.
    assert(decide(300, Seq(record(totalBytes = 900L * GiB))) ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NotDownward(2000, 300)))
  }

  test("AE7: a normal value below one wave is left alone") {
    // Even a one-byte stage cannot pull the recommendation under a single full wave.
    assert(decide(200, Seq(record(totalBytes = 1L))) ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NotDownward(2000, 200)))
  }

  test("the default reduction factor imposes no threshold but an override still blocks") {
    // A 3000 -> 2000 reduction is not 2x, so v1's default would have blocked it.
    decide(3000, Seq(record(totalBytes = 900L * GiB))) match {
      case applied: DownwardShuffleDecision.Applied => assert(applied.selectedValue == 2000)
      case other => fail(s"expected an applied reduction but got $other")
    }
    val strictConfig = baseConfig.copy(minReductionFactor = 2.0)
    assert(decide(3000, Seq(record(totalBytes = 900L * GiB)), strictConfig) ==
      DownwardShuffleDecision.Skipped(
        DownwardShuffleSkipReason.BelowReductionThreshold(2000, 3000, 2.0)))
    // Exactly 2x still clears the explicit override.
    assert(decide(4000, Seq(record(totalBytes = 900L * GiB)), strictConfig)
      .isInstanceOf[DownwardShuffleDecision.Applied])
  }

  test("AE7: the candidate is always a whole wave, never above the normal value") {
    val configs = Seq(baseConfig, baseConfig.copy(targetPartitionSizeBytes = 512L * 1024L * 1024L),
      baseConfig.copy(inputSizeFactor = 0.8, minReductionFactor = 2.0))
    val byteSizes = Seq(0L, 1L, GiB, 37L * GiB, 999L * GiB, 100000L * GiB, Long.MaxValue)
    val slotCounts = Seq(1, 3, 16, 375, 2000, 100000)
    val normalValues = Seq(1, 199, 200, 500, 501, 2000, 200000, Int.MaxValue)
    for (config <- configs; bytes <- byteSizes; slotCount <- slotCounts;
         normalValue <- normalValues) {
      decide(normalValue, Seq(record(totalBytes = bytes)), config, slotCount = Some(slotCount))
      match {
        case applied: DownwardShuffleDecision.Applied =>
          val context = s"config=$config bytes=$bytes slots=$slotCount normal=$normalValue"
          assert(applied.selectedValue < normalValue, context)
          assert(applied.selectedValue >= slotCount, context)
          assert(applied.selectedValue % slotCount == 0, context)
          assert(applied.selectedValue == applied.waveCount * slotCount, context)
        case _ => // a skip is always safe
      }
    }
  }

  //
  // Evidence completeness
  //

  test("incomplete evidence keeps the normal recommendation and warns") {
    val incomplete = ShuffleStageInputAnalysis(
      Seq(record(totalBytes = 900L * GiB)),
      Seq(ShuffleStageInputIncompleteReason.MissingExchangeMetric(0L, 3L, "Exchange")),
      ShuffleInputProvenance.Measured)
    DownwardShufflePartitionsPolicy.decide(Right(baseConfig), 4000, Some(slots),
      incomplete) match {
      case DownwardShuffleDecision.Skipped(reason: DownwardShuffleSkipReason.IncompleteEvidence) =>
        assert(reason.isWarning)
        assert(reason.description.contains("data size"))
      case other => fail(s"expected an incomplete-evidence skip but got $other")
    }
  }

  test("an analysis that was never produced fails closed without a user comment") {
    val notAnalyzed = ShuffleStageInputAnalysis.empty(ShuffleInputProvenance.Measured)
    assert(!notAnalyzed.isComplete)
    val decision =
      DownwardShufflePartitionsPolicy.decide(Right(baseConfig), 4000, Some(slots), notAnalyzed)
    assert(decision ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NoAnalysisAvailable))
    // A provider that never produced an analysis is not something the user can act on.
    assert(!DownwardShuffleSkipReason.NoAnalysisAvailable.isWarning)
  }

  test("an application with no shuffle at all is a quiet no-op") {
    val noShuffle = ShuffleStageInputAnalysis(Seq.empty, Seq.empty, ShuffleInputProvenance.Measured)
    assert(noShuffle.isComplete)
    val decision =
      DownwardShufflePartitionsPolicy.decide(Right(baseConfig), 4000, Some(slots), noShuffle)
    assert(decision == DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NoStageEvidence))
    assert(!DownwardShuffleSkipReason.NoStageEvidence.isWarning)
  }

  test("the incomplete summary stays concise") {
    val reasons = (1 to 5).map { i =>
      ShuffleStageInputIncompleteReason.UnresolvedConsumerStage(0L, i.toLong, "Exchange")
    }
    val summary = ShuffleStageInputAnalysis(Seq.empty, reasons, ShuffleInputProvenance.Measured)
      .incompleteSummary()
    assert(summary.contains("and 3 more"))
    assert(summary.contains("node 1"))
    assert(summary.contains("node 2"))
    assert(!summary.contains("node 3"))
  }

  //
  // Diagnostics
  //

  test("the applied comment names the slot count and wave arithmetic, not rung rounding") {
    val stage = record(sqlId = 2L, stageId = 7, stageAttemptId = 1,
      totalBytes = 4500L * GiB, numBranches = 2)
    val applied = decide(8000, Seq(stage)).asInstanceOf[DownwardShuffleDecision.Applied]
    assert(applied.selectedValue == 6000 && applied.waveCount == 3)
    val comment = DownwardShufflePartitionsPolicy.appliedComment(
      Seq("spark.sql.shuffle.partitions",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum"), applied)
    Seq("spark.sql.shuffle.partitions",
      "spark.sql.adaptive.coalescePartitions.initialPartitionNum",
      "8000", "6000", "measured", "SQL 2", "stage 7", "attempt 1",
      (4500L * GiB).toString, "2 shuffle branch(es)", "1.0", "4500",
      "3 execution wave(s)", "2000 cluster task slots").foreach { fragment =>
      assert(comment.contains(fragment), s"'$fragment' missing from: $comment")
    }
    assert(!comment.contains("rung"), s"the comment should no longer mention rungs: $comment")
  }
}
