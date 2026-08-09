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

package com.nvidia.spark.rapids.tool.tuning.config

import com.nvidia.spark.rapids.tool.ToolTestUtils
import com.nvidia.spark.rapids.tool.tuning.TuningEntry
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

class PySparkMemoryTuningPolicySuite extends AnyFunSuite {
  private def profilingPolicy(
      default: List[TuningConfigEntry],
      profiling: List[TuningConfigEntry]): PySparkMemoryTuningPolicy = {
    val overrides = ToolTestUtils.buildTuningConfigs(default = default, profiling = profiling)
    val provider = TuningConfigProvider.builder
      .withUserProvidedConfig(Some(overrides))
      .build[ProfTuningConfigProvider]
    PySparkMemoryTuningPolicy.from(provider)
  }

  test("PySpark memory policy uses settled defaults") {
    val policy = PySparkMemoryTuningPolicy.from(
      TuningConfigProvider.builder.build[ProfTuningConfigProvider])

    policy.evidenceHeadroomMultiplier shouldBe BigDecimal("1.25")
    policy.retryGrowthFactor shouldBe BigDecimal("1.5")
    policy.rebalanceSource shouldBe PySparkMemoryRebalanceSource.Heap
  }

  test("PySpark memory policy honors default and tool-specific overrides") {
    val policy = profilingPolicy(
      default = List(
        TuningConfigEntry(
          name = PySparkMemoryTuningPolicy.EVIDENCE_HEADROOM_MULTIPLIER,
          default = "1.40"),
        TuningConfigEntry(
          name = PySparkMemoryTuningPolicy.REBALANCE_SOURCE,
          default = "OVERHEAD")),
      profiling = List(TuningConfigEntry(
        name = PySparkMemoryTuningPolicy.RETRY_GROWTH_FACTOR,
        default = "1.75")))

    policy.evidenceHeadroomMultiplier shouldBe BigDecimal("1.40")
    policy.retryGrowthFactor shouldBe BigDecimal("1.75")
    policy.rebalanceSource shouldBe PySparkMemoryRebalanceSource.Overhead
  }

  test("PySpark memory multiplier validation rejects invalid values") {
    Seq("not-a-number", "NaN", "Infinity", "-Infinity", "1", "0", "-1").foreach { value =>
      val error = the[IllegalArgumentException] thrownBy profilingPolicy(
        default = List(TuningConfigEntry(
          name = PySparkMemoryTuningPolicy.EVIDENCE_HEADROOM_MULTIPLIER,
          default = value)),
        profiling = List.empty)
      error.getMessage shouldBe
        s"Invalid ${PySparkMemoryTuningPolicy.EVIDENCE_HEADROOM_MULTIPLIER} value '$value': " +
          "expected a finite decimal value greater than 1."
    }
  }

  test("PySpark memory retry multiplier uses the same strict validation") {
    val value = "1.0"
    val error = the[IllegalArgumentException] thrownBy profilingPolicy(
      default = List(TuningConfigEntry(
        name = PySparkMemoryTuningPolicy.RETRY_GROWTH_FACTOR,
        default = value)),
      profiling = List.empty)
    error.getMessage shouldBe
      s"Invalid ${PySparkMemoryTuningPolicy.RETRY_GROWTH_FACTOR} value '$value': " +
        "expected a finite decimal value greater than 1."
  }

  test("PySpark memory rebalance source parsing is case-insensitive") {
    PySparkMemoryRebalanceSource.parse("heap") shouldBe PySparkMemoryRebalanceSource.Heap
    PySparkMemoryRebalanceSource.parse("OvErHeAd") shouldBe
      PySparkMemoryRebalanceSource.Overhead
  }

  test("PySpark memory rebalance source rejects unknown values") {
    val value = "offheap"
    val error = the[IllegalArgumentException] thrownBy profilingPolicy(
      default = List(TuningConfigEntry(
        name = PySparkMemoryTuningPolicy.REBALANCE_SOURCE,
        default = value)),
      profiling = List.empty)
    error.getMessage shouldBe
      s"Invalid ${PySparkMemoryTuningPolicy.REBALANCE_SOURCE} value '$value': " +
        "expected HEAP or OVERHEAD."
  }

  test("PySpark executor memory is an enabled cluster bootstrap byte setting") {
    val definition = TuningEntryDefinition
      .getEntryDefinition(PySparkMemoryTuningPolicy.PYSPARK_MEMORY_KEY).get

    definition.isEnabled() shouldBe true
    definition.getLevelAsEnum shouldBe LevelEnum.Cluster
    definition.getConfTypeAsEnum shouldBe ConfTypeEnum.Byte
    definition.getConfUnit shouldBe Some("MiB")
    definition.isBootstrap() shouldBe true

    val entry = TuningEntry.build(
      PySparkMemoryTuningPolicy.PYSPARK_MEMORY_KEY, Some("4096"), None, Some(definition))
    entry.getOriginalValue shouldBe Some("4g")
    entry.isBootstrap() shouldBe true
  }
}
