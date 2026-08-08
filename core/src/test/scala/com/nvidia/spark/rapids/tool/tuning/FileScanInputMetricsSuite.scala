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

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.AppSummaryInfoBaseProvider

class FileScanInputMetricsSuite extends ProfilingAutoTunerSuiteBase {
  test("base provider reports no reliable file scan input") {
    assert(new AppSummaryInfoBaseProvider().getMaxFileScanInput.isEmpty)
  }

  test("AutoTuner test provider can represent an absent reliable input") {
    val provider = getMockInfoProvider(
      maxInput = 3292825256.0,
      spilledMetrics = Seq.empty,
      jvmGCFractions = Seq.empty,
      propsFromLog = mutable.Map.empty[String, String],
      sparkVersion = Some(testSparkVersion),
      maxFileScanInputOverride = Some(None))
    assert(provider.getMaxFileScanInput.isEmpty)
  }
}
