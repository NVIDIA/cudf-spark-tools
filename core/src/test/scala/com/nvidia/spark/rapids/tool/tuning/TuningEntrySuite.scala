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
import com.nvidia.spark.rapids.tool.tuning.config.{ConfTypeEnum, MissingCommentPolicy,
  ProfTuningConfigProvider, QualTuningConfigProvider, TuningConfigEntry, TuningConfigProvider,
  TuningEntryDefinition}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.sql.rapids.tool.InvalidMemoryUnitFormatException

class TuningEntrySuite extends AnyFunSuite {
  private val runtimeBatchSizeKey = GpuBatchAndConcurrencyKeys.BatchSizeBytes

  private def memoryDefinition(
      specialValues: Seq[String]): TuningEntryDefinition = {
    TuningEntryDefinition(
      label = "test.memory",
      confType = ConfTypeEnum.Byte,
      defaultUnit = Some("Byte"),
      specialValues = specialValues)
  }

  test("memory entry preserves a declared special value") {
    val entry = TuningEntry.build(
      "test.memory", Some("-1"), None, Some(memoryDefinition(Seq("-1"))))
    entry.getOriginalValue shouldBe Some("-1")
  }

  test("memory entry normalizes negative one when it is not declared special") {
    val entry = TuningEntry.build(
      "test.memory", Some("-1"), None, Some(memoryDefinition(Seq.empty)))
    entry.getOriginalValue shouldBe Some("-1b")
  }

  test("memory entry rejects an undeclared negative value") {
    an[InvalidMemoryUnitFormatException] should be thrownBy {
      TuningEntry.build(
        "test.memory", Some("-2g"), None, Some(memoryDefinition(Seq("-1"))))
    }
  }

  test("AQE broadcast threshold declares negative one as special") {
    val definition = TuningEntryDefinition
      .getEntryDefinition("spark.sql.adaptive.autoBroadcastJoinThreshold").get
    definition.isSpecialValue("-1") shouldBe true
  }

  test("cache serializer configuration is loaded from YAML") {
    val configProvider = TuningConfigProvider.builder.build[ProfTuningConfigProvider]
    configProvider.getEntry(AutoTuner.CACHE_SERIALIZER_CONFIG).getDefault shouldBe
      "com.nvidia.spark.ParquetCachedBatchSerializer"

    val definition = TuningEntryDefinition
      .getEntryDefinition(AutoTuner.CACHE_SERIALIZER_PROPERTY).get
    definition.getDefaultSpark shouldBe
      "org.apache.spark.sql.execution.columnar.DefaultCachedBatchSerializer"
    definition.isBootstrap() shouldBe true
  }

  test("cache serializer definition lookup is safe when the entry is unavailable") {
    AutoTuner.getCacheSerializerDefinition(Map.empty) shouldBe None
  }

  test("profiling provider recognizes a named override from the profiling section") {
    val config = ToolTestUtils.buildTuningConfigs(profiling = List(
      TuningConfigEntry(name = "READER_MULTITHREADED_COMBINE_THRESHOLD", default = "32m")))
    val provider = TuningConfigProvider.builder
      .withUserProvidedConfig(Some(config))
      .build[ProfTuningConfigProvider]

    provider.getUserProvidedDefault("READER_MULTITHREADED_COMBINE_THRESHOLD") shouldBe Some("32m")
    provider.isDefaultValueUserProvided("READER_MULTITHREADED_COMBINE_THRESHOLD") shouldBe true
  }

  test("qualification provider recognizes a named override from the qualification section") {
    val config = ToolTestUtils.buildTuningConfigs(qualification = List(
      TuningConfigEntry(name = "READER_MULTITHREADED_COMBINE_THRESHOLD", default = "32m")))
    val provider = TuningConfigProvider.builder
      .withUserProvidedConfig(Some(config))
      .build[QualTuningConfigProvider]

    provider.getUserProvidedDefault("READER_MULTITHREADED_COMBINE_THRESHOLD") shouldBe Some("32m")
    provider.isDefaultValueUserProvided("READER_MULTITHREADED_COMBINE_THRESHOLD") shouldBe true
  }

  test("concurrent GPU tasks omits its missing-property comment") {
    val definition = TuningEntryDefinition
      .getEntryDefinition(GpuBatchAndConcurrencyKeys.ConcurrentGpuTasks).get

    definition.getMissingCommentPolicy shouldBe MissingCommentPolicy.Omit
  }

  test("missing-property comments use the default policy when it is not configured") {
    val definition = TuningEntryDefinition("test.property")

    definition.getMissingCommentPolicy shouldBe MissingCommentPolicy.Default
  }

  test("unknown missing-comment policies are rejected") {
    val definition = TuningEntryDefinition("test.property")
    definition.getComments.put("missingPolicy", "unexpected")

    val error = intercept[IllegalArgumentException] {
      definition.getMissingCommentPolicy
    }
    error.getMessage shouldBe "Unknown missing comment policy: unexpected"
  }

  test("runtime byte entry accepts surrounding whitespace using Spark's parser") {
    val definition = TuningEntryDefinition.getEntryDefinition(runtimeBatchSizeKey)
    val entry = new RuntimeByteTuningEntry(
      runtimeBatchSizeKey, Some(" 2G "), None, definition)

    entry.getOriginalValue shouldBe Some("2g")
  }

  test("runtime byte entry accepts zero and retains invalid negative values") {
    val definition = TuningEntryDefinition.getEntryDefinition(runtimeBatchSizeKey)
    val entry = new RuntimeByteTuningEntry(
      runtimeBatchSizeKey, Some("0"), Some("-1g"), definition)

    entry.getOriginalValue shouldBe Some("0b")
    entry.tunedValue shouldBe Some("-1g")
  }

  test("runtime byte entry retains invalid values and accepts a later valid target") {
    val definition = TuningEntryDefinition.getEntryDefinition(runtimeBatchSizeKey)
    val entry = new RuntimeByteTuningEntry(
      runtimeBatchSizeKey, Some("1GiB"), Some("1.5g"), definition)

    entry.getOriginalValue shouldBe Some("1GiB")
    entry.tunedValue shouldBe Some("1.5g")

    entry.setRecommendedValue(" 2G ")
    entry.tunedValue shouldBe Some("2147483648b")
    entry.getTuneValue() shouldBe "2g"
  }
}
