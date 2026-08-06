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

  private def profProvider(
      default: List[TuningConfigEntry] = List.empty): ProfTuningConfigProvider = {
    TuningConfigProvider.builder
      .withUserProvidedConfig(Some(ToolTestUtils.buildTuningConfigs(default = default)))
      .build[ProfTuningConfigProvider]
  }

  private def qualProvider(
      default: List[TuningConfigEntry] = List.empty,
      qualification: List[TuningConfigEntry] = List.empty): QualTuningConfigProvider = {
    TuningConfigProvider.builder
      .withUserProvidedConfig(Some(
        ToolTestUtils.buildTuningConfigs(default = default, qualification = qualification)))
      .build[QualTuningConfigProvider]
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

  /** Config used by the arithmetic tests: 1 GiB target, rungs 500/1000/2000/..., 2x reduction. */
  private val baseConfig = DownwardShufflePolicyConfig(
    enabled = true,
    targetPartitionSizeBytes = GiB,
    inputSizeFactor = 1.0,
    partitionFloor = 500,
    rungMultiplier = 2.0,
    minReductionFactor = 2.0)

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
      provenance: ShuffleInputProvenance = ShuffleInputProvenance.Measured
  ): DownwardShuffleDecision = {
    DownwardShufflePartitionsPolicy.decide(Right(config), normalValue,
      analysis(records, provenance))
  }

  //
  // Configuration
  //

  test("profiling defaults match the product contract") {
    val config = expectValid(DownwardShufflePolicyConfig.fromProvider(profProvider()))
    assert(config.enabled)
    assert(config.targetPartitionSizeBytes == GiB)
    assert(config.inputSizeFactor == 1.0)
    assert(config.partitionFloor == 500)
    assert(config.rungMultiplier == 2.0)
    assert(config.minReductionFactor == 2.0)
  }

  test("qualification overrides the input size factor in its own tool section") {
    val config = expectValid(DownwardShufflePolicyConfig.fromProvider(qualProvider()))
    assert(config.enabled)
    assert(config.inputSizeFactor == 0.8)
    // Everything else still comes from the shared defaults.
    assert(config.targetPartitionSizeBytes == GiB)
    assert(config.partitionFloor == 500)
  }

  test("user overrides are honored in the default and tool sections") {
    val fromDefault = DownwardShufflePolicyConfig.fromProvider(profProvider(
      default = List(
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_TARGET_PARTITION_SIZE", default = "512m"),
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_PARTITION_FLOOR", default = "128"))))
    assert(fromDefault == Right(baseConfig.copy(
      targetPartitionSizeBytes = 512L * 1024L * 1024L, partitionFloor = 128)))

    val fromToolSection = DownwardShufflePolicyConfig.fromProvider(qualProvider(
      qualification = List(
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_INPUT_SIZE_FACTOR", default = "0.5"))))
    assert(fromToolSection == Right(baseConfig.copy(inputSizeFactor = 0.5)))
  }

  test("a disabled feature short-circuits without validating the other entries") {
    val provider = profProvider(default = List(
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_ENABLED", default = "false"),
      // Deliberately invalid; it must not be read while the feature is off.
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_PARTITION_FLOOR", default = "not-a-number")))
    val configResult = DownwardShufflePolicyConfig.fromProvider(provider)
    assert(configResult == Right(DownwardShufflePolicyConfig.disabled))
    assert(DownwardShufflePartitionsPolicy.decide(configResult, 4000,
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
      "DOWNWARD_SHUFFLE_PARTITION_FLOOR" -> Seq("0", "-500", "1.5", "abc"),
      // A multiplier of exactly 1.0 or below cannot generate growing rungs.
      "DOWNWARD_SHUFFLE_RUNG_MULTIPLIER" -> Seq("1", "1.0", "0.5", "abc"),
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
    Seq("DOWNWARD_SHUFFLE_INPUT_SIZE_FACTOR", "DOWNWARD_SHUFFLE_RUNG_MULTIPLIER",
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
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_PARTITION_FLOOR", default = "-1"),
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_RUNG_MULTIPLIER", default = "0.5")))
    val configResult = DownwardShufflePolicyConfig.fromProvider(provider)
    val errors = expectErrors(configResult)
    assert(errors.size == 2)
    DownwardShufflePartitionsPolicy.decide(configResult, 4000,
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

  test("rungAtOrAbove generates each rung from the floor and the multiplier") {
    val rungs = Seq(1L -> 500, 500L -> 500, 501L -> 1000, 1000L -> 1000, 1001L -> 2000,
      2000L -> 2000, 2001L -> 4000, 4001L -> 8000, 64001L -> 128000)
    rungs.foreach { case (requirement, expected) =>
      assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(requirement, 500, 2.0) ==
        Some(expected), s"requirement $requirement")
    }
  }

  test("fractional multipliers round each rung up so the sequence stays integral") {
    // 100 -> 150 -> 225 -> 338 (ceil of 337.5) -> 507
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(101L, 100, 1.5) == Some(150))
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(226L, 100, 1.5) == Some(338))
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(339L, 100, 1.5) == Some(507))
  }

  test("rung generation always makes progress") {
    // Without the forced +1 step, a multiplier this close to 1.0 would never advance.
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(505L, 500, 1.0000001) == Some(505))
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(505L, 500, 1.0) == Some(505))
  }

  test("a requirement no representable rung can cover fails closed") {
    // Beyond the largest partition count Spark can express.
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(Int.MaxValue.toLong + 1L, 500, 2.0)
      .isEmpty)
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(Long.MaxValue, 500, 2.0).isEmpty)
    // Representable, but the doubling sequence overshoots Int.MaxValue before reaching it.
    assert(DownwardShufflePartitionsPolicy.rungAtOrAbove(Int.MaxValue.toLong, 500, 2.0).isEmpty)
  }

  test("an out-of-range requirement skips instead of recommending an uncovering rung") {
    decide(Int.MaxValue, Seq(record(totalBytes = Long.MaxValue))) match {
      case DownwardShuffleDecision.Skipped(
          reason @ DownwardShuffleSkipReason.RequirementOutOfRange(_)) =>
        assert(reason.isWarning)
      case other => fail(s"expected an out-of-range skip but got $other")
    }
  }

  //
  // Policy decisions
  //

  test("AE1: a multi-input join stage is sized from its combined branches") {
    // Two branches whose combined input needs 730 partitions at a 1 GiB target.
    val joinStage = record(sqlId = 0L, stageId = 7, totalBytes = 729L * GiB + 1L, numBranches = 2)
    decide(normalValue = 2000, records = Seq(joinStage)) match {
      case applied: DownwardShuffleDecision.Applied =>
        assert(applied.rawRequirement == 730L)
        assert(applied.selectedValue == 1000)
        assert(applied.normalValue == 2000)
        assert(applied.determiningRecord == joinStage)
        assert(applied.provenance == ShuffleInputProvenance.Measured)
      case other => fail(s"expected an applied reduction but got $other")
    }
  }

  test("the qualification factor lowers the requirement of the same stage") {
    val stage = record(totalBytes = 1000L * GiB)
    val qualConfig = baseConfig.copy(inputSizeFactor = 0.8)
    decide(4000, Seq(stage), qualConfig, ShuffleInputProvenance.Estimated) match {
      case applied: DownwardShuffleDecision.Applied =>
        assert(applied.estimatedInputBytes == 800L * GiB)
        assert(applied.rawRequirement == 800L)
        assert(applied.selectedValue == 1000)
        assert(applied.inputSizeFactor == 0.8)
        assert(applied.provenance == ShuffleInputProvenance.Estimated)
      case other => fail(s"expected an applied reduction but got $other")
    }
  }

  test("the worst consumer stage determines the candidate") {
    val small = record(sqlId = 0L, stageId = 1, totalBytes = 100L * GiB)
    val worst = record(sqlId = 1L, stageId = 9, totalBytes = 1500L * GiB)
    Seq(Seq(small, worst), Seq(worst, small)).foreach { records =>
      decide(8000, records) match {
        case applied: DownwardShuffleDecision.Applied =>
          assert(applied.determiningRecord == worst)
          assert(applied.selectedValue == 2000)
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

  test("an exact target multiple does not round up to an extra rung") {
    decide(4000, Seq(record(totalBytes = 1000L * GiB))) match {
      case applied: DownwardShuffleDecision.Applied =>
        assert(applied.rawRequirement == 1000L)
        assert(applied.selectedValue == 1000)
      case other => fail(s"expected an applied reduction but got $other")
    }
  }

  test("AE7: a candidate at or above the normal value never raises it") {
    // The candidate rung equals the current value.
    assert(decide(1000, Seq(record(totalBytes = 900L * GiB))) ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NotDownward(1000, 1000)))
    // The candidate rung is larger than the current value.
    assert(decide(300, Seq(record(totalBytes = 900L * GiB))) ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NotDownward(1000, 300)))
  }

  test("AE7: a normal value below the rung floor is left alone") {
    // Even a tiny stage cannot pull the recommendation under the floor of 500.
    assert(decide(200, Seq(record(totalBytes = 1L))) ==
      DownwardShuffleDecision.Skipped(DownwardShuffleSkipReason.NotDownward(500, 200)))
  }

  test("AE7: a reduction below the minimum factor is not applied") {
    // 1999 is downward from a 1000-partition candidate but is not a 2x reduction.
    assert(decide(1999, Seq(record(totalBytes = 900L * GiB))) ==
      DownwardShuffleDecision.Skipped(
        DownwardShuffleSkipReason.BelowReductionThreshold(1000, 1999, 2.0)))
    // Exactly 2x is enough.
    assert(decide(2000, Seq(record(totalBytes = 900L * GiB)))
      .isInstanceOf[DownwardShuffleDecision.Applied])
  }

  test("the calculator never returns a value above the normal one or below the floor") {
    val configs = Seq(baseConfig, baseConfig.copy(partitionFloor = 128, rungMultiplier = 1.5),
      baseConfig.copy(inputSizeFactor = 0.8, minReductionFactor = 1.0))
    val byteSizes = Seq(0L, 1L, GiB, 37L * GiB, 999L * GiB, 100000L * GiB, Long.MaxValue)
    val normalValues = Seq(1, 199, 200, 500, 501, 2000, 200000, Int.MaxValue)
    for (config <- configs; bytes <- byteSizes; normalValue <- normalValues) {
      decide(normalValue, Seq(record(totalBytes = bytes)), config) match {
        case applied: DownwardShuffleDecision.Applied =>
          assert(applied.selectedValue < normalValue,
            s"config=$config bytes=$bytes normal=$normalValue")
          assert(applied.selectedValue >= config.partitionFloor,
            s"config=$config bytes=$bytes normal=$normalValue")
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
    DownwardShufflePartitionsPolicy.decide(Right(baseConfig), 4000, incomplete) match {
      case DownwardShuffleDecision.Skipped(reason: DownwardShuffleSkipReason.IncompleteEvidence) =>
        assert(reason.isWarning)
        assert(reason.description.contains("data size"))
      case other => fail(s"expected an incomplete-evidence skip but got $other")
    }
  }

  test("an analysis that was never produced is incomplete, not empty-but-complete") {
    val notAnalyzed = ShuffleStageInputAnalysis.empty(ShuffleInputProvenance.Measured)
    assert(!notAnalyzed.isComplete)
    DownwardShufflePartitionsPolicy.decide(Right(baseConfig), 4000, notAnalyzed) match {
      case DownwardShuffleDecision.Skipped(reason: DownwardShuffleSkipReason.IncompleteEvidence) =>
        assert(reason.description.contains(ShuffleStageInputAnalysis.notAnalyzedSummary))
      case other => fail(s"expected an incomplete-evidence skip but got $other")
    }
  }

  test("an application with no shuffle at all is a quiet no-op") {
    val noShuffle = ShuffleStageInputAnalysis(Seq.empty, Seq.empty, ShuffleInputProvenance.Measured)
    assert(noShuffle.isComplete)
    val decision = DownwardShufflePartitionsPolicy.decide(Right(baseConfig), 4000, noShuffle)
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

  test("the applied comment names every input of the decision") {
    val stage = record(sqlId = 2L, stageId = 7, stageAttemptId = 1,
      totalBytes = 729L * GiB + 1L, numBranches = 2)
    val applied = decide(2000, Seq(stage)).asInstanceOf[DownwardShuffleDecision.Applied]
    val comment = DownwardShufflePartitionsPolicy.appliedComment(
      Seq("spark.sql.shuffle.partitions",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum"), applied)
    Seq("spark.sql.shuffle.partitions",
      "spark.sql.adaptive.coalescePartitions.initialPartitionNum",
      "2000", "1000", "measured", "SQL 2", "stage 7", "attempt 1",
      (729L * GiB + 1L).toString, "2 shuffle branch(es)", "1.0", "730").foreach { fragment =>
      assert(comment.contains(fragment), s"'$fragment' missing from: $comment")
    }
  }
}
