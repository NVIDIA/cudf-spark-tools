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

import com.nvidia.spark.rapids.tool.{PlatformFactory, PlatformNames, ToolTestUtils}
import com.nvidia.spark.rapids.tool.tuning.config.TuningConfigEntry

class PageablePoolSizingSuite extends BaseAutoTunerSuite {
  val autoTunerHelper: AutoTunerHelper = ProfilingAutoTunerHelper

  private def recommendations(
      pool: Option[String] = None, cap: String = "4g"): Map[String, String] = {
    val props = mutable.LinkedHashMap(
      "spark.executor.cores" -> "16",
      "spark.executor.instances" -> "2",
      "spark.sql.shuffle.partitions" -> "1000",
      "spark.executor.memory" -> "32g",
      "spark.executor.memoryOverhead" -> "12g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.task.resource.gpu.amount" -> "0.001",
      "spark.rapids.memory.host.offHeapLimit.enabled" -> "true",
      "spark.rapids.memory.pinnedPool.size" -> "4506m",
      "spark.rapids.sql.concurrentGpuTasks" -> "3",
      "spark.rapids.sql.enabled" -> "true",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    pool.foreach(props.put("spark.rapids.memory.pageablePool.size", _))
    val enforced = props.toMap - "spark.executor.instances" -
      "spark.rapids.memory.pageablePool.size" - "spark.sql.shuffle.partitions"
    val target = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16), memoryGB = Some(64L), gpuCount = Some(1),
      gpuDevice = Some("l4"), enforcedSparkProperties = enforced)
    val platform = PlatformFactory.createInstance(PlatformNames.EMR, Some(target))
    configureEventLogClusterInfoForTest(platform, numCores = 16, numWorkers = 2,
      sparkProperties = props.toMap)
    val provider = getMockInfoProvider(0, Seq(0), Seq(0.0), props, Some(testSparkVersion))
    val config = ToolTestUtils.buildTuningConfigs(default = List(
      TuningConfigEntry(name = "PAGEABLE_POOL_MAX_SIZE", default = cap)))
    buildAutoTunerForTests(provider, platform, Some(Yarn), Some(config))
      .getRecommendedProperties(showOnlyUpdatedProps = false)._1
      .map(p => p.name -> p.getTuneValue()).toMap
  }

  test("integrated pool sizing preserves explicit zero and nonzero sizes") {
    val key = "spark.rapids.memory.pageablePool.size"
    assert(recommendations().get(key).contains("2355m"))
    val zero = recommendations(pool = Some("0")).get(key)
    assert(zero.forall(v => v == "0" || v == "0m" || v == "0b"), zero.toString)
    assert(recommendations(pool = Some("4g")).get(key).forall(v => v == "4g" || v == "4096m"))
  }

  test("pageable pool leaves headroom, respects a host cap, and avoids tiny residual pools") {
    assert(PageablePoolSizing.sizeMiB(12288, 4506, 4096, None).contains(2355L))
    assert(PageablePoolSizing.sizeMiB(12288, 4506, 4096, Some(6000)).contains(747L))
    assert(PageablePoolSizing.sizeMiB(4096, 3000, 4096, None).isEmpty)
    assert(PageablePoolSizing.sizeMiB(12288, 4506, 0, None).isEmpty)
    assert(PageablePoolSizing.sizeMiB(65536, 1024, 4096, None).contains(4096L))
  }

  test("pool recommendation is disabled by default") {
    assert(!recommendations(cap = "0").contains("spark.rapids.memory.pageablePool.size"))
  }
}
