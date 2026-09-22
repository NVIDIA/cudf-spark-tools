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

package com.nvidia.spark.rapids.tool

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

class PlatformTargetGpuDeviceMemoryCapacitySuite extends AnyFunSuite {
  private def createPlatform(
      platformName: String,
      targetCluster: com.nvidia.spark.rapids.tool.tuning.TargetClusterProps): Platform = {
    targetCluster.validate()
    PlatformFactory.createInstance(platformName, Some(targetCluster))
  }

  test("OnPrem target GPU capacity keeps its raw value and source") {
    val target = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16),
      memoryGB = Some(64L),
      gpuCount = Some(1),
      gpuMemory = Some(" 20G "),
      gpuDevice = Some(GpuTypes.L4))

    val platform = createPlatform(PlatformNames.ONPREM, target)

    platform.recommendedTargetGpuDeviceMemoryCapacitySpec shouldBe
      TargetGpuDeviceMemoryCapacitySpec(
        " 20G ", TargetGpuCapacitySource.ExplicitTargetConfig)
  }

  test("CSP target instance keeps explicitly supplied GPU capacity") {
    val target = ToolTestUtils.buildTargetClusterInfo(
      workerNodeInstanceType = Some("g2-standard-8"),
      gpuMemory = Some("20g"))

    val platform = createPlatform(PlatformNames.DATAPROC, target)

    platform.recommendedTargetGpuDeviceMemoryCapacitySpec shouldBe
      TargetGpuDeviceMemoryCapacitySpec("20g", TargetGpuCapacitySource.ExplicitTargetConfig)
  }

  test("CSP with OnPrem-style target resources keeps explicitly supplied GPU capacity") {
    val target = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16),
      memoryGB = Some(64L),
      gpuCount = Some(1),
      gpuMemory = Some("20g"),
      gpuDevice = Some(GpuTypes.L4))

    val platform = createPlatform(PlatformNames.DATAPROC, target)

    platform.recommendedTargetGpuDeviceMemoryCapacitySpec shouldBe
      TargetGpuDeviceMemoryCapacitySpec("20g", TargetGpuCapacitySource.ExplicitTargetConfig)
  }

  test("missing target GPU capacity uses the device catalog") {
    val target = ToolTestUtils.buildTargetClusterInfo(
      workerNodeInstanceType = Some("g2-standard-8"))

    val platform = createPlatform(PlatformNames.DATAPROC, target)

    platform.recommendedTargetGpuDeviceMemoryCapacitySpec shouldBe
      TargetGpuDeviceMemoryCapacitySpec(
        L4Gpu.getMemory, TargetGpuCapacitySource.DeviceCatalog)
  }
}
