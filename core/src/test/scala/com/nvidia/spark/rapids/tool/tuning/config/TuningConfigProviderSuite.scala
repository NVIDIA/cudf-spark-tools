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
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

class TuningConfigProviderSuite extends AnyFunSuite {
  private val BatchSizeBytes = "BATCH_SIZE_BYTES"
  private val ConcGpuTasks = "CONC_GPU_TASKS"
  private val ReaderMultithreadedCombineThreshold = "READER_MULTITHREADED_COMBINE_THRESHOLD"

  private def profilingProvider(
      default: List[TuningConfigEntry] = List.empty,
      profiling: List[TuningConfigEntry] = List.empty,
      qualification: List[TuningConfigEntry] = List.empty): ProfTuningConfigProvider = {
    val overrides = ToolTestUtils.buildTuningConfigs(
      default = default, qualification = qualification, profiling = profiling)
    TuningConfigProvider.builder
      .withUserProvidedConfig(Some(overrides))
      .build[ProfTuningConfigProvider]
  }

  test("shipped maximum is not reported as user-provided") {
    val provider = TuningConfigProvider.builder.build[ProfTuningConfigProvider]

    provider.getEntry(ConcGpuTasks).getMax shouldBe "4"
    provider.getUserProvidedMax(ConcGpuTasks) shouldBe None
  }

  test("shipped default is not reported as user-provided") {
    val provider = TuningConfigProvider.builder.build[ProfTuningConfigProvider]

    provider.getEntry(ReaderMultithreadedCombineThreshold).getDefault shouldBe "10m"
    provider.getUserProvidedDefault(ReaderMultithreadedCombineThreshold) shouldBe None
  }

  test("user default retains provenance") {
    val provider = profilingProvider(default = List(
      TuningConfigEntry(name = ReaderMultithreadedCombineThreshold, default = "32m")))

    provider.getUserProvidedDefault(ReaderMultithreadedCombineThreshold) shouldBe Some("32m")
  }

  test("shipped tool-specific default masks a user default") {
    val provider = profilingProvider(
      default = List(TuningConfigEntry(name = BatchSizeBytes, default = "3g")))

    provider.getEntry(BatchSizeBytes).getDefault shouldBe "2147483647"
    provider.getUserProvidedDefault(BatchSizeBytes) shouldBe None
  }

  test("user tool-specific default retains active provenance") {
    val provider = profilingProvider(
      default = List(TuningConfigEntry(name = BatchSizeBytes, default = "3g")),
      profiling = List(TuningConfigEntry(name = BatchSizeBytes, default = "1g")))

    provider.getEntry(BatchSizeBytes).getDefault shouldBe "1g"
    provider.getUserProvidedDefault(BatchSizeBytes) shouldBe Some("1g")
  }

  test("user default maximum retains provenance even when equal to shipped value") {
    val provider = profilingProvider(
      default = List(TuningConfigEntry(name = ConcGpuTasks, max = "4")))

    provider.getUserProvidedMax(ConcGpuTasks) shouldBe Some("4")
  }

  test("active tool-specific maximum retains user provenance") {
    val provider = profilingProvider(
      default = List(TuningConfigEntry(name = ConcGpuTasks, max = "8")),
      profiling = List(TuningConfigEntry(name = ConcGpuTasks, max = "6")))

    provider.getEntry(ConcGpuTasks).getMax shouldBe "6"
    provider.getUserProvidedMax(ConcGpuTasks) shouldBe Some("6")
  }

  test("inactive tool-specific maximum does not affect profiling provenance") {
    val provider = profilingProvider(
      qualification = List(TuningConfigEntry(name = ConcGpuTasks, max = "7")))

    provider.getEntry(ConcGpuTasks).getMax shouldBe "4"
    provider.getUserProvidedMax(ConcGpuTasks) shouldBe None
  }

  test("qualification provider retains its active tool-specific maximum provenance") {
    val overrides = ToolTestUtils.buildTuningConfigs(
      qualification = List(TuningConfigEntry(name = ConcGpuTasks, max = "7")))
    val provider = TuningConfigProvider.builder
      .withUserProvidedConfig(Some(overrides))
      .build[QualTuningConfigProvider]

    provider.getUserProvidedMax(ConcGpuTasks) shouldBe Some("7")
  }

  test("user entry without a maximum does not claim maximum provenance") {
    val provider = profilingProvider(
      default = List(TuningConfigEntry(name = ConcGpuTasks, default = "3")))

    provider.getEntry(ConcGpuTasks).getMax shouldBe "4"
    provider.getUserProvidedMax(ConcGpuTasks) shouldBe None
  }
}
