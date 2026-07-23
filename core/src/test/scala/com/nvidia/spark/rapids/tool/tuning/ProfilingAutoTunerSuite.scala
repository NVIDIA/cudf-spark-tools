/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

import java.io.{File, FileNotFoundException}

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.{A100Gpu, AppSummaryInfoBaseProvider, GpuDevice, NodeInstanceMapKey, Platform, PlatformFactory, PlatformInstanceTypes, PlatformNames, T4Gpu}
import com.nvidia.spark.rapids.tool.ToolTestUtils
import com.nvidia.spark.rapids.tool.planparser.db.DBVersionExtractor
import com.nvidia.spark.rapids.tool.profiling.{DriverLogUnsupportedOperators, ProfileArgs, ProfileMain, Profiler}
import org.scalatest.prop.TableDrivenPropertyChecks._
import org.scalatest.prop.TableFor4

import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.rapids.tool.util.FSUtils
import org.apache.spark.sql.rapids.tool.util.WebCrawlerUtil

/**
 * Base class for Profiling AutoTuner test suites.
 */
abstract class ProfilingAutoTunerSuiteBase extends BaseAutoTunerSuite {

  val autoTunerHelper: AutoTunerHelper = ProfilingAutoTunerHelper
  val profilingLogDir: String = ToolTestUtils.getTestResourcePath("spark-events-profiling")

  /**
   * Helper method to get the path to the output file
   */
  protected def getOutputFilePath(outputDir: File, fileName: String): String = {
    val profilerOutputDir = new File(outputDir, s"${Profiler.SUBDIR}")
    val outputFile = profilerOutputDir.listFiles()
      .filter(_.isDirectory)
      .flatMap(dir => dir.listFiles().filter(file => file.isFile && file.getName == fileName))
      .headOption

    outputFile match {
      case Some(file) => file.getAbsolutePath
      case None => throw new FileNotFoundException(
        s"File $fileName not found in ${profilerOutputDir.getAbsolutePath}")
    }
  }

  /**
   * Helper method to extract the AutoTuner results from the profile log content
   * TODO: We should store the AutoTuner results in a separate file.
   */
  protected def extractAutoTunerResults(profileLogContent: String): String = {
    val startSubstring = "### D. Recommended Configuration ###"
    val indexOfAutoTunerOutput = profileLogContent.indexOf(startSubstring)
    if (indexOfAutoTunerOutput > 0) {
      profileLogContent.substring(indexOfAutoTunerOutput + startSubstring.length).trim
    } else {
      ""
    }
  }

  protected def getGpuAppMockInfoProvider(
      maxInput: Double = 0,
      spilledMetrics: Seq[Long] = Seq(0),
      jvmGCFractions: Seq[Double] = Seq(0.0),
      propsFromLog: mutable.Map[String, String] = mutable.Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"),
      rapidsJars: Seq[String] = Seq()): AppInfoProviderMockTest = {
    getMockInfoProvider(maxInput, spilledMetrics, jvmGCFractions, propsFromLog,
      Some(testSparkVersion), rapidsJars)
  }

  protected def getGpuAppMockInfoWithJars(rapidsJars: Seq[String]): AppInfoProviderMockTest = {
    getMockInfoProvider(0, Seq(0), Seq(0.0),
      mutable.Map("spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin"),
      Some(testSparkVersion), rapidsJars)
  }

  /**
   * Helper method to return an instance of the Profiling AutoTuner with default properties
   * for Dataproc.
   */
  protected def buildDefaultDataprocAutoTuner(
      logEventsProps: mutable.Map[String, String]) : AutoTuner = {
    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "212992MiB")

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      sparkPropsWithMemory, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    buildAutoTunerForTests(infoProvider, platform)
  }

  /**
   * Helper method to return the latest Spark RAPIDS plugin jar URL.
   */
  protected lazy val latestPluginJarUrl: String = {
    val latestRelease = WebCrawlerUtil.getLatestPluginRelease match {
      case Some(v) => v
      case None => fail("Could not find pull the latest release successfully")
    }
    ToolTestUtils.pluginMvnPrefix(latestRelease) + ".jar"
  }
}

/**
 * Test suite for the Profiling Tool's AutoTuner (DEPRECATED. Use [[ProfilingAutoTunerSuiteV2]])
 *
 * IMPORTANT NOTE:
 * 1. This test suite uses the legacy worker info properties format, which is overloaded to be
 *    used for both source and target cluster properties.
 * 2. These tests will be migrated to use the new target cluster info format, which explicitly
 *    specifies target cluster shape and Spark properties.
 * 3. All new Profiling AutoTuner test cases should be added to [[ProfilingAutoTunerSuiteV2]]
 *    instead of this suite.
 *
 * TODO:
 * Migrate all tests in this suite to use the new target cluster properties format.
 * https://github.com/NVIDIA/spark-rapids-tools/issues/1748
 */
class ProfilingAutoTunerSuite extends ProfilingAutoTunerSuiteBase {

  test("verify 3.2.0+ auto conf setting") {
    val sparkProps = Map(
      "spark.rapids.sql.enabled" -> "true",
      "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.executor.memory" -> "122880MiB")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      mutable.Map() ++ sparkProps, Some("3.2.0"), Seq())
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 2,
      gpuCount = 1,
      sparkProperties = sparkProps
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=2
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark320.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.cores' was not set.
          |- 'spark.executor.instances' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Load non-existing cluster properties from event log") {
    val platform = PlatformFactory.createInstance()
    val autoTuner = ProfilingAutoTunerHelper
      .buildAutoTuner(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.locality.wait=0
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.files.maxPartitionBytes=512m
          |
          |Comments:
          |- 'spark.executor.cores' should be set to 16.
          |- 'spark.executor.instances' should be set to (cpuCoresPerNode * numWorkers) / 'spark.executor.cores'.
          |- 'spark.executor.memory' should be set to 2g/core.
          |- 'spark.rapids.memory.pinnedPool.size' should be set to 2g.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' should be set to Min(4, (gpuMemory / 7500m)).
          |- 'spark.rapids.sql.enabled' should be true to enable SQL operations on the GPU.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.enabled' should be enabled for better performance.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' should be set to 0.001.
          |- Could not infer the cluster configuration, recommendations are generated using default values!
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  ignore("Load cluster properties from event log with CPU cores 0") {
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 0,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1")
    )

    val autoTuner = buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=8
          |--conf spark.locality.wait=0
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.cores' was not set.
          |- 'spark.executor.instances' was not set.
          |- 'spark.executor.memory' should be set to 2g/core.
          |- 'spark.rapids.memory.pinnedPool.size' should be set to 2g.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  ignore("Load cluster properties from event logs with memory to cores ratio to small") {
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.memory" -> "14000MiB"
      )
    )

    val autoTuner = buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=[FILL_IN_VALUE]
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.cores' was not set.
          |- 'spark.executor.instances' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${notEnoughMemCommentForKey("spark.executor.memory")}
          |- ${notEnoughMemCommentForKey("spark.rapids.memory.pinnedPool.size")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- ${notEnoughMemComment(40140)}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  ignore("Load cluster properties from event logs with CPU memory missing") {
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.master" -> "local")
    )

    val autoTuner = buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=[FILL_IN_VALUE]
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.cores' was not set.
          |- 'spark.executor.instances' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- Not enough memory to set 'spark.executor.memory'. See comments for more details.
          |- Not enough memory to set 'spark.rapids.memory.pinnedPool.size'. See comments for more details.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- ${notEnoughMemComment(40140)}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  ignore("Load cluster properties from event logs with CPU memory 0") {
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform = platform,
      sparkProperties = Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.memory" -> "0m"
      )
    )

    val autoTuner = buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=4
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.cores' was not set.
          |- 'spark.executor.instances' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  ignore("Load cluster properties from event logs with number of workers 0") {
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 0,
      gpuCount = 2,
      sparkProperties = Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.memory" -> "122880MiB"
      )
    )

    val autoTuner = buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.locality.wait=0
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.files.maxPartitionBytes=512m
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.cores' should be set to 16.
          |- 'spark.executor.instances' should be set to (cpuCoresPerNode * numWorkers) / 'spark.executor.cores'.
          |- 'spark.executor.memory' should be set to 2g/core.
          |- 'spark.rapids.memory.pinnedPool.size' should be set to 2g.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' should be set to Min(4, (gpuMemory / 7500m)).
          |- 'spark.rapids.sql.enabled' should be true to enable SQL operations on the GPU.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' should be set to 0.001.
          |- Could not infer the cluster configuration, recommendations are generated using default values!
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test should ignore executor memory specified in the properties as
  // CSP nodes have fixed memory configurations.
  // TODO: Revisit this test
  test("Test executor memory on CSP where executor memory/cpu ratio is small") {
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 4,
      gpuCount = 1,
      sparkProperties = Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.memory" -> "15360MiB"
      )
    )

    val autoTuner = buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=4
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.cores' was not set.
          |- 'spark.executor.instances' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("test AutoTuner with empty sparkProperties") {
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2
    )
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.cores' was not set.
          |- 'spark.executor.instances' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    val autoTuner =
      buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("AutoTuner detects non UTF-8 file-encoding") {
    // When system properties has an entry for file-encoding that is not supported by GPU for
    // certain expressions. Then the AutoTuner should generate a comment warning that the
    // file-encoding is not one of the supported ones.
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "212992MiB",
      "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
      "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
      "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
      "spark.shuffle.manager" ->
        s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        // set the file-encoding to non UTF-8
        "file.encoding" -> "ISO-8859-1",
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4")
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val combinedProps = customProps ++ logEventsProps
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- file.encoding should be [UTF-8] because GPU only supports the charset when using some expressions.
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }
  // Test that the properties from the custom props will be used to calculate the recommendations.
  // For example, the output should use "spark.sql.shuffle.partitions" -> 400 when calculating the
  // recommendations instead of the one from the event log which is 200.
  test("AutoTuner gives precedence to properties from custom props") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "4",
      "spark.executor.memory" -> "212992MiB",
      "spark.sql.shuffle.partitions" -> "400",
      "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
      "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
      "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
      "spark.shuffle.manager" ->
        s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4")
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    // customProps should take precedence over logEventsProps per test purpose
    val combinedProps = logEventsProps ++ customProps
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test verifies that AutoTuner recommends the correct value for
  // "spark.plugins" property.
  test("test 'spark.plugins' is recommended correctly") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.concurrentGpuTasks" -> "4")
    val combinedProps = customProps ++ logEventsProps
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), combinedProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test verifies that Profiling AutoTuner retains existing
  // "spark.plugins" property and RAPIDS plugin is added to it.
  test("test existing 'spark.plugins' are retained and RAPIDS plugin is added") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "212992MiB",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.plugins" -> "com.nvidia.spark.WrongPlugin0, com.nvidia.spark.WrongPlugin1",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.concurrentGpuTasks" -> "4",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh")
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val combinedProps = customProps ++ logEventsProps
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.WrongPlugin0,com.nvidia.spark.WrongPlugin1,com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.plugins' should include the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test that the AutoTuner recommends the config for enabling
  // the plugin when the plugin is not enabled.
  test("plugin not enabled") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "212992MiB",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "false",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4")
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val combinedProps = customProps ++ logEventsProps
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Changing the maxInput of tasks should reflect on the maxPartitions recommendations.
  // Values used in setting the properties are taken from sample eventlogs.
  test("Recommendation of maxPartitions is calculated based on maxInput of tasks") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "212992MiB")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
        "spark.sql.shuffle.partitions" -> "1000",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1")
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val combinedProps = customProps ++ logEventsProps
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(
      getMockInfoProvider(3.7449728E7, Seq(0, 0), Seq(0.01, 0.0), logEventsProps,
        Some(testSparkVersion)), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=1000
          |--conf spark.sql.files.maxPartitionBytes=3669m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // When GCFraction is higher AutoTuner.MAX_JVM_GCTIME_FRACTION, the output should contain
  // a comment recommending to consider different GC algorithms.
  // This test triggers that case by injecting a sequence of jvmGCFraction with average higher
  // than the static threshold of 0.3.
  test("Output contains GC comments when GC Fraction is higher than threshold") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "212992MiB")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
        "spark.sql.shuffle.partitions" -> "1000",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1")
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(0, 0), Seq(0.4, 0.4), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val combinedProps = customProps ++ logEventsProps
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=1000
          |--conf spark.sql.files.maxPartitionBytes=3669m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Ensure handling of integer values for maxPartitionBytes is handled
  test("Handle integer value of maxPartitionBytes properly") {
    val customProps = mutable.LinkedHashMap(
      "spark.sql.files.maxPartitionBytes" -> "12345678")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
        "spark.sql.shuffle.partitions" -> "1000",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1")
    val combinedProps = logEventsProps ++ customProps
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(0, 0), Seq(0.4, 0.4),
      combinedProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=1000
          |--conf spark.sql.files.maxPartitionBytes=39m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  def generateRecommendationsForRapidsJars(rapidsJars: Seq[String]): String = {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "16",
      "spark.executor.memory" -> "122880MiB",
      "spark.executor.memoryOverhead" -> "8396m",
      "spark.rapids.memory.pinnedPool.size" -> "4096m",
      "spark.rapids.shuffle.multiThreaded.reader.threads" -> "16",
      "spark.rapids.shuffle.multiThreaded.writer.threads" -> "16",
      "spark.rapids.sql.concurrentGpuTasks" -> "3",
      "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
      "spark.shuffle.manager" ->
        s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
      "spark.sql.files.maxPartitionBytes" -> "512m",
      "spark.task.resource.gpu.amount" -> "0.001")
    val sparkProps = defaultDataprocProps.++(customProps)
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = sparkProps.toMap
    )

    val autoTuner =
      buildAutoTunerForTests(getGpuAppMockInfoProvider(
        propsFromLog = sparkProps,
        rapidsJars = rapidsJars), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    Profiler.getAutoTunerResultsAsString(properties, comments)
  }

  test("Multiple RAPIDS jars trigger a comment") {
    // 1. The Autotuner should warn the users that they have multiple jars defined in the classPath
    // 2. Compare the output
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.multiple")} [23.06.0, 23.02.1]
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    val rapidsJarsArr = Seq("rapids-4-spark_2.12-23.06.0-SNAPSHOT.jar",
      "rapids-4-spark_2.12-23.02.1.jar")
    val autoTunerOutput = generateRecommendationsForRapidsJars(rapidsJarsArr)
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Recommend upgrading to the latest plugin release") {
    // 1. Pull the latest release from mvn.
    // 2. The Autotuner should warn the users that they are using an older release
    // 3. Compare the output
    val testAppJarVer = "23.02.0"
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${latestPluginJarComment(latestPluginJarUrl, testAppJarVer)}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    val rapidsJarsArr = Seq(s"rapids-4-spark_2.12-$testAppJarVer.jar")
    val autoTunerOutput = generateRecommendationsForRapidsJars(rapidsJarsArr)
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Helper that runs the AutoTuner without pre-setting `spark.rapids.sql.concurrentGpuTasks`
  // so the default recommendation path is exercised. Returns the AutoTuner output string.
  private def runConcurrentGpuTasksScenario(
      rapidsJars: Seq[String],
      enforcedProps: Map[String, String] = Map.empty,
      preserveProps: List[String] = List.empty): String = {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "16",
      "spark.executor.memory" -> "122880MiB",
      "spark.executor.memoryOverhead" -> "8396m",
      "spark.rapids.memory.pinnedPool.size" -> "4096m",
      "spark.rapids.shuffle.multiThreaded.reader.threads" -> "16",
      "spark.rapids.shuffle.multiThreaded.writer.threads" -> "16",
      "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
      "spark.shuffle.manager" ->
        s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
      "spark.sql.files.maxPartitionBytes" -> "512m",
      "spark.task.resource.gpu.amount" -> "0.001")
    val sparkProps = defaultDataprocProps.++(customProps)
    val platform = if (enforcedProps.nonEmpty || preserveProps.nonEmpty) {
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        enforcedSparkProperties = enforcedProps,
        preserveSparkProperties = preserveProps
      )
      PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))
    } else {
      PlatformFactory.createInstance(PlatformNames.DATAPROC)
    }
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = sparkProps.toMap
    )
    val autoTuner =
      buildAutoTunerForTests(getGpuAppMockInfoProvider(
        propsFromLog = sparkProps,
        rapidsJars = rapidsJars), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    Profiler.getAutoTunerResultsAsString(properties, comments)
  }

  test("AutoTuner drops concurrentGpuTasks recommendation for plugin >= 25.06") {
    val output = runConcurrentGpuTasksScenario(Seq("rapids-4-spark_2.12-25.06.0.jar"))
    assert(!output.contains("spark.rapids.sql.concurrentGpuTasks"),
      s"Expected no concurrentGpuTasks recommendation/comment, got:\n$output")
  }

  test("AutoTuner keeps concurrentGpuTasks recommendation for plugin < 25.06") {
    val output = runConcurrentGpuTasksScenario(Seq("rapids-4-spark_2.12-25.04.0.jar"))
    assert(output.contains("spark.rapids.sql.concurrentGpuTasks"),
      s"Expected concurrentGpuTasks to be present, got:\n$output")
  }

  test("AutoTuner keeps concurrentGpuTasks recommendation when no plugin jar version found") {
    val output = runConcurrentGpuTasksScenario(Seq.empty)
    assert(output.contains("spark.rapids.sql.concurrentGpuTasks"),
      s"Expected concurrentGpuTasks to be present, got:\n$output")
  }

  test("Target cluster enforced concurrentGpuTasks overrides plugin >= 25.06 drop") {
    val output = runConcurrentGpuTasksScenario(
      Seq("rapids-4-spark_2.12-25.08.0.jar"),
      enforcedProps = Map("spark.rapids.sql.concurrentGpuTasks" -> "4"))
    assert(output.contains("spark.rapids.sql.concurrentGpuTasks=4"),
      s"Expected enforced concurrentGpuTasks=4 to be present, got:\n$output")
  }

  test("Target cluster preserve concurrentGpuTasks overrides plugin >= 25.06 drop") {
    val output = runConcurrentGpuTasksScenario(
      Seq("rapids-4-spark_2.12-25.08.0.jar"),
      preserveProps = List("spark.rapids.sql.concurrentGpuTasks"))
    assert(output.contains("spark.rapids.sql.concurrentGpuTasks"),
      s"Expected preserved concurrentGpuTasks to be present, got:\n$output")
  }

  test("No recommendation when the jar pluginJar is up-to-date") {
    // 1. Pull the latest release from mvn.
    // 2. The Autotuner finds tha the jar version is latest. No comments should be added
    // 3. Compare the output
    val latestRelease = WebCrawlerUtil.getLatestPluginRelease match {
      case Some(v) => v
      case None => fail("Could not find pull the latest release successfully")
    }
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    val rapidsJarsArr = Seq(s"rapids-4-spark_2.12-$latestRelease.jar")
    val autoTunerOutput = generateRecommendationsForRapidsJars(rapidsJarsArr)
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Note: This test verifies that the AutoTuner comments about enabling the file cache
  // but does not actually enable since this requires knowledge of the disk bandwidth
  // and available disk space.
  test("Comment about enabling file cache if parquet/orc and data thresholds are met") {
    val customProps = mutable.LinkedHashMap(
      "spark.sql.files.maxPartitionBytes" -> "12345678")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
        "spark.sql.shuffle.partitions" -> "1000",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1")
    val combinedProps = logEventsProps ++ customProps
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(0, 0), Seq(0.4, 0.4), combinedProps,
      Some(testSparkVersion), Seq(), 40.0, 200000000000L)
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.filecache.enabled=false
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=1000
          |--conf spark.sql.files.maxPartitionBytes=39m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.filecache.enabled' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- Enable file cache only if Spark local disks bandwidth is > 1 GB/s and you have sufficient disk space available to fit both cache and normal Spark temporary data.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Do not recommend file cache if small redundant size") {
    val customProps = mutable.LinkedHashMap(
      "spark.sql.files.maxPartitionBytes" -> "12345678")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "1m",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
        "spark.sql.shuffle.partitions" -> "1000",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1")
    val combinedProps = logEventsProps ++ customProps
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(0, 0), Seq(0.4, 0.4), combinedProps,
      Some(testSparkVersion), Seq(), 40.0, 2000000L)
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=1000
          |--conf spark.sql.files.maxPartitionBytes=39m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("test recommendations for databricks-aws platform argument") {
    val platform = PlatformFactory.createInstance(PlatformNames.DATABRICKS_AWS)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = Map(
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.memory" -> "122880MiB" // default
      )
    )

    val autoTuner = buildAutoTunerForTests(getGpuAppMockInfoProvider(), platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()

    // Assert recommendations are excluded in properties
    assert(properties.map(_.name).forall(autoTuner.platform.isValidRecommendation))
    // Assert recommendations are skipped in comments
    assert(comments.map(_.comment).forall(autoTuner.platform.isValidComment))
  }

  // Test cases for memory overhead configuration based on Spark Master
  private val MEMORY_OVERHEAD_TEST_CASES = Table(
    ("sparkMaster", "shouldIncludeMemoryOverhead"),
    // memoryOverhead should be included for yarn and k8s
    (Some(Yarn), true),
    (Some(Kubernetes), true),
    // memoryOverhead should be excluded for standalone and local
    (Some(Standalone), false),
    (Some(Local), false),
    // memoryOverhead should be excluded for other sparkMaster values
    (None, false)
  )

  forAll(MEMORY_OVERHEAD_TEST_CASES) {
    (sparkMaster: Option[SparkMaster], shouldIncludeMemoryOverhead: Boolean) =>
      test("memoryOverhead comment is " +
        s"${if (shouldIncludeMemoryOverhead) "included" else "excluded"} " +
        s"for sparkMaster=${sparkMaster.getOrElse("undefined")}") {
        val logEventsProps = mutable.LinkedHashMap[String, String](
          "spark.executor.cores" -> "16",
          "spark.executor.memory" -> "80g",
          "spark.executor.resource.gpu.amount" -> "1",
          "spark.executor.instances" -> "1",
          "spark.sql.shuffle.partitions" -> "200",
          "spark.sql.files.maxPartitionBytes" -> "1g",
          "spark.task.resource.gpu.amount" -> "0.001",
          "spark.rapids.memory.pinnedPool.size" -> "5g",
          "spark.rapids.sql.enabled" -> "true",
          "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
          "spark.rapids.sql.concurrentGpuTasks" -> "4"
        )

        val infoProvider = getMockInfoProvider(8126464.0, Seq(0),
          Seq(0.004), logEventsProps, Some(testSparkVersion))
        val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

        val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "122880MiB")
        configureEventLogClusterInfoForTest(
          platform,
          numCores = 32,
          numWorkers = 4,
          gpuCount = 2,
          sparkProperties = sparkPropsWithMemory.toMap
        )

        val autoTuner = buildAutoTunerForTests(infoProvider, platform, sparkMaster)
        val (_, comments) = autoTuner.getRecommendedProperties()
        val expectedComment = "'spark.executor.memoryOverhead' was not set."

        if (shouldIncludeMemoryOverhead) {
          assert(comments.exists(_.comment == expectedComment),
            s"Expected comment '$expectedComment' not found")
        } else {
          assert(comments.forall(_.comment != expectedComment),
            s"Unexpected comment '$expectedComment' found")
        }
      }
  }

  test("Recommendations generated for unsupported operators from driver logs only") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m",
      "spark.rapids.sql.concurrentGpuTasks" -> "3",
      "spark.task.resource.gpu.amount" -> "0.001")
    val unsupportedDriverOperators = Seq(
      DriverLogUnsupportedOperators(
        "FromUnixTime", 1,
        "Only UTC zone id is supported. Actual default zone id: America/Los_Angeles; " +
          "CORRECTED format 'yyyyMMdd' on the GPU is not guaranteed to produce the same " +
          "results as Spark on CPU. Set spark.rapids.sql.incompatibleDateFormats.enabled=true " +
          "to force onto GPU.")
    )
    val driverInfoProvider = DriverInfoProviderMockTest(unsupportedDriverOperators)
    val platform = PlatformFactory.createInstance(PlatformNames.DEFAULT)

    // Configure cluster info using Platform's existing method
    val sparkPropsWithMemory = customProps + ("spark.executor.memory" -> "122880MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    val autoTuner: AutoTuner = ProfilingAutoTunerHelper
      .buildAutoTunerFromProps(AppSummaryInfoBaseProvider.fromAppInfo(None),
        platform, driverInfoProvider)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.rapids.sql.incompatibleDateFormats.enabled=true
          |
          |Comments:
          |- 'spark.rapids.sql.incompatibleDateFormats.enabled' was not set.
          |- ${commentForExperimentalConfig("spark.rapids.sql.incompatibleDateFormats.enabled")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Recommendations generated for unsupported operators from driver and event logs") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m",
      "spark.rapids.sql.concurrentGpuTasks" -> "2",
      "spark.task.resource.gpu.amount" -> "0.001")
    val unsupportedDriverOperators = Seq(
      DriverLogUnsupportedOperators(
        "FromUnixTime", 1,
        "Only UTC zone id is supported. Actual default zone id: America/Los_Angeles; " +
          "CORRECTED format 'yyyyMMdd' on the GPU is not guaranteed to produce the same " +
          "results as Spark on CPU. Set spark.rapids.sql.incompatibleDateFormats.enabled=true " +
          "to force onto GPU.")
    )
    val driverInfoProvider = DriverInfoProviderMockTest(unsupportedDriverOperators)
    val platform = PlatformFactory.createInstance(PlatformNames.DEFAULT)

    // Configure cluster info using Platform's existing method

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = customProps.toMap
    )

    val autoTuner: AutoTuner = ProfilingAutoTunerHelper
      .buildAutoTunerFromProps(
        getGpuAppMockInfoProvider(propsFromLog = customProps), platform, driverInfoProvider)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    // SourceCoresPreservingStrategy preserves source executor.cores=8 (from spark props),
    // so memory and thread calculations are based on 8 cores instead of platform-recommended 16.
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=16g
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.incompatibleDateFormats.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |
          |Comments:
          |- 'spark.executor.instances' was not set.
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.incompatibleDateFormats.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- ${commentForExperimentalConfig("spark.rapids.sql.incompatibleDateFormats.enabled")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Recommendations generated for empty unsupported operators from driver logs only") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "122880MiB",
      "spark.rapids.sql.concurrentGpuTasks" -> "3",
      "spark.task.resource.gpu.amount" -> "0.001")
    val platform = PlatformFactory.createInstance(PlatformNames.DEFAULT)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = customProps.toMap
    )

    val autoTuner: AutoTuner = ProfilingAutoTunerHelper
      .buildAutoTunerFromProps(AppSummaryInfoBaseProvider.fromAppInfo(None),
        platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|Cannot recommend properties. See Comments.
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Recommendations not generated for unsupported operators from driver logs") {
    // This test does not generate any recommendations for the unsupported operator 'Literal'
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m",
      "spark.rapids.sql.concurrentGpuTasks" -> "3",
      "spark.task.resource.gpu.amount" -> "0.001")
    val unsupportedDriverOperators = Seq(
      DriverLogUnsupportedOperators(
        "Literal", 3,
        "expression Literal 1700518632630000 produces an unsupported type TimestampType")
    )
    val driverInfoProvider = DriverInfoProviderMockTest(unsupportedDriverOperators)
    val platform = PlatformFactory.createInstance(PlatformNames.DEFAULT)

    // Configure cluster info using Platform's existing method
    val sparkPropsWithMemory = customProps + ("spark.executor.memory" -> "122880MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    val autoTuner: AutoTuner = ProfilingAutoTunerHelper
      .buildAutoTunerFromProps(AppSummaryInfoBaseProvider.fromAppInfo(None),
        platform, driverInfoProvider)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|Cannot recommend properties. See Comments.
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("AQE configuration autoBroadcastJoinThreshold should not be GTE 100mb") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.executor.memoryOverhead" -> "7372m",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4",
        "spark.sql.adaptive.enabled" -> "true",
        "spark.sql.adaptive.autoBroadcastJoinThreshold" -> "500mb")
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val combinedProps = customProps ++ logEventsProps + ("spark.executor.memory" -> "212992MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- Setting 'spark.sql.adaptive.autoBroadcastJoinThreshold' > 100m could lead to performance\n  regression. Should be set to a lower number.
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  private def testPartitionConfigurations(
                                           inputSize: Double,
                                           shuffleRead: Double,
                                           gpuDevice: GpuDevice,
                                           expectedLines: Seq[String]): Unit = {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m",
      "spark.rapids.sql.concurrentGpuTasks" -> "3",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4")
    val combinedProps = logEventsProps ++ customProps
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), combinedProps,
      Some(testSparkVersion), meanInput = inputSize, meanShuffleRead = shuffleRead)
    val targetClusterProps = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(32),
      memoryGB = Some(0L),
      gpuCount = Some(4),
      gpuMemory = Some(gpuDevice.getMemory),
      gpuDevice = Some(gpuDevice.toString)
    )
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterProps))

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    assert(expectedLines.forall(line => autoTunerOutput.contains(line)),
      s"Expected lines not found in AutoTuner output")
  }

  val testCases: TableFor4[Int, Int, GpuDevice, Seq[String]] = Table(
    ("inputSize", "shuffleRead", "gpuDevice", "expectedLines"),
    // small input, small shuffle read
    (1000, 1000, T4Gpu, Seq(
      "--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m")),
    // large input, small shuffle read
    (40000, 1000, T4Gpu, Seq()),
    // large input, large shuffle read
    (40000, 80000, T4Gpu, Seq(
      "--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=32m",
      "--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=800",
      "--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false"
    )),
    // large input, large shuffle read, faster GPU
    (40000, 80000, A100Gpu, Seq(
      "--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=64m",
      "--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400",
      "--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false"
    ))
  )

  forAll(testCases) { (inputSize, shuffleRead, gpuDevice, expectedLines) =>
    test(s"AQE partition configs - input size: $inputSize," +
      s" shuffle read: $shuffleRead, gpu device: $gpuDevice") {
      testPartitionConfigurations(inputSize, shuffleRead, gpuDevice, expectedLines)
    }
  }

  test("Handle adaptive auto shuffle configuration setting properly") {
    val customProps = mutable.LinkedHashMap(
      "spark.databricks.adaptive.autoOptimizeShuffle.enabled" -> "true")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.rapids.sql.multiThreadedRead.numThreads" -> "20",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager",
        "spark.sql.shuffle.partitions" -> "1000",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1")
    val combinedProps = logEventsProps ++ customProps
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(0, 0), Seq(0.4, 0.4), combinedProps,
      Some(testDatabricksVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATABRICKS_AWS)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.databricks.adaptive.autoOptimizeShuffle.enabled=false
          |--conf spark.executor.memory=64g
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6553m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=48
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=48
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=64
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersionDatabricks.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=1000
          |--conf spark.sql.files.maxPartitionBytes=3669m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  /**
   * Helper method to verify that the recommended shuffle manager version matches the
   * expected version.
   */
  private def verifyRecommendedShuffleManagerVersion(
                                                      autoTuner: AutoTuner,
                                                      expectedSmVersion: String): Unit = {
    autoTuner.getShuffleManagerClassName match {
      case Right(smClassName) =>
        assert(smClassName == ProfilingAutoTunerHelper
          .buildShuffleManagerClassName(expectedSmVersion))
      case Left(comment) =>
        fail(s"Expected valid RapidsShuffleManager but got comment: $comment")
    }
  }

  val dbPlatform: Platform = PlatformFactory.createInstance(PlatformNames.DATABRICKS_AWS)
  dbPlatform.supportedShuffleManagerVersionMap.foreach { case (dbVersion, smVersion) =>
    test(s"test shuffle manager version for supported databricks version - $dbVersion") {
      val databricksVersion = s"$dbVersion.x-gpu-ml-scala2.12"
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
        mutable.Map("spark.rapids.sql.enabled" -> "true",
          "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
          DBVersionExtractor.DB_SPARK_VERSION_KEY -> databricksVersion),
        Some(databricksVersion), Seq())
      val autoTuner = buildAutoTunerForTests(
        infoProvider,
        PlatformFactory.createInstance(PlatformNames.DATABRICKS_AWS))
      // Assert shuffle manager string for given Databricks version
      verifyRecommendedShuffleManagerVersion(autoTuner, expectedSmVersion = smVersion)
    }
  }

  val sparkPlatform: Platform = PlatformFactory.createInstance(PlatformNames.DEFAULT)
  sparkPlatform.supportedShuffleManagerVersionMap.foreach { case (sparkVersion, smVersion) =>
    test(s"test shuffle manager version for supported spark version - $sparkVersion") {
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
        mutable.Map("spark.rapids.sql.enabled" -> "true",
          "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin"),
        Some(sparkVersion), Seq())
      val autoTuner = buildAutoTunerForTests(infoProvider, PlatformFactory.createInstance())
      // Assert shuffle manager string for given Spark version
      verifyRecommendedShuffleManagerVersion(autoTuner, expectedSmVersion = smVersion)
    }
  }

  test("test shuffle manager version for supported custom spark version") {
    val customSparkVersion = "3.3.0-custom"
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      mutable.Map("spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin"),
      Some(customSparkVersion), Seq())
    val autoTuner = buildAutoTunerForTests(infoProvider, PlatformFactory.createInstance())
    // Assert shuffle manager string for supported custom Spark v3.3.0
    verifyRecommendedShuffleManagerVersion(autoTuner, expectedSmVersion = "330")
  }

  /**
   * Helper method to verify that the shuffle manager version is not recommended
   * for the unsupported Spark version.
   */
  private def verifyUnsupportedSparkVersionForShuffleManager(
                                                              autoTuner: AutoTuner,
                                                              sparkVersion: String): Unit = {
    autoTuner.getShuffleManagerClassName match {
      case Right(smClassName) =>
        fail(s"Expected error comment but got valid RapidsShuffleManager: $smClassName")
      case Left(comment) =>
        assert(comment == shuffleManagerCommentForUnsupportedVersion(sparkVersion,
          autoTuner.platform))
    }
  }

  test("test shuffle manager version for unsupported databricks version") {
    val databricksVersion = "9.1.x-gpu-ml-scala2.12"
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      mutable.Map("spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin",
        DBVersionExtractor.DB_SPARK_VERSION_KEY -> databricksVersion),
      Some(databricksVersion), Seq())
    // Do not set the platform as DB to see if it can work correctly irrespective
    val autoTuner = buildAutoTunerForTests(
      infoProvider,
      PlatformFactory.createInstance(PlatformNames.DATABRICKS_AWS))
    verifyUnsupportedSparkVersionForShuffleManager(autoTuner, databricksVersion)
  }

  test("test shuffle manager version for unsupported spark version") {
    val sparkVersion = "3.1.2"
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      mutable.Map("spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin"),
      Some(sparkVersion), Seq())
    val autoTuner = buildAutoTunerForTests(infoProvider, PlatformFactory.createInstance())
    verifyUnsupportedSparkVersionForShuffleManager(autoTuner, sparkVersion)
  }

  test("test shuffle manager version for unsupported custom spark version") {
    val customSparkVersion = "3.1.2-custom"
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      mutable.Map("spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin"),
      Some(customSparkVersion), Seq())
    val autoTuner = buildAutoTunerForTests(infoProvider, PlatformFactory.createInstance())
    verifyUnsupportedSparkVersionForShuffleManager(autoTuner, customSparkVersion)
  }

  test("test shuffle manager version for missing spark version") {
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      mutable.Map("spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.AnotherPlugin, com.nvidia.spark.SQLPlugin"),
      None, Seq())
    val autoTuner = buildAutoTunerForTests(infoProvider, PlatformFactory.createInstance())
    // Verify that the shuffle manager is not recommended for missing Spark version
    autoTuner.getShuffleManagerClassName match {
      case Right(smClassName) =>
        fail(s"Expected error comment but got valid RapidsShuffleManager: $smClassName")
      case Left(comment) =>
        assert(comment == shuffleManagerCommentForMissingVersion)
    }
  }

  test("Test spilling occurred in shuffle stages") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m",
      "spark.task.resource.gpu.amount" -> "0.001")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager")
    val combinedProps = logEventsProps ++ customProps
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(1000L, 1000L), Seq(0.4, 0.4),
      combinedProps, Some(testSparkVersion), shuffleStagesWithPosSpilling = Set(1))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
          |--conf spark.sql.files.maxPartitionBytes=3669m
          |--conf spark.sql.shuffle.partitions=400
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- ${classPathComments("rapids.jars.missing")}
          |- $shufflePartitionsCommentForSpilling
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Test spilling occurred in shuffle stages with shuffle skew") {
    val customProps = mutable.LinkedHashMap(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "47222m")
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4",
        "spark.rapids.shuffle.multiThreaded.reader.threads" -> "8",
        "spark.rapids.shuffle.multiThreaded.writer.threads" -> "8",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.shuffle.manager" ->
          s"com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager")
    val combinedProps = logEventsProps ++ customProps
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(1000L, 1000L), Seq(0.4, 0.4),
      combinedProps, Some(testSparkVersion), shuffleStagesWithPosSpilling = Set(1, 5),
      shuffleSkewStages = Set(1))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = combinedProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.files.maxPartitionBytes=3669m
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- ${classPathComments("rapids.jars.missing")}
          |- Shuffle skew exists (when task's Shuffle Read Size > 3 * Avg Stage-level size) in
          |  stages with spilling. Increasing shuffle partitions is not recommended in this
          |  case since keys will still hash to the same task.
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Test Kryo Serializer sets Registrar") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val autoTuner = buildDefaultDataprocAutoTuner(logEventsProps)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.kryo.registrator=com.nvidia.spark.rapids.GpuKryoRegistrator
          |--conf spark.kryoserializer.buffer.max=512m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.kryo.registrator' should include GpuKryoRegistrator when using Kryo serialization.
          |- 'spark.kryoserializer.buffer.max' increasing the max buffer to prevent out-of-memory errors.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Test Kryo Serializer sets Registrar when already set") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.registrator" -> "org.apache.SomeRegistrator,org.apache.SomeOtherRegistrator",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val autoTuner = buildDefaultDataprocAutoTuner(logEventsProps)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.kryo.registrator=org.apache.SomeRegistrator,org.apache.SomeOtherRegistrator,com.nvidia.spark.rapids.GpuKryoRegistrator
          |--conf spark.kryoserializer.buffer.max=512m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.kryo.registrator' GpuKryoRegistrator must be appended to the existing value when using Kryo serialization.
          |- 'spark.kryoserializer.buffer.max' increasing the max buffer to prevent out-of-memory errors.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Test Kryo Serializer sets Registrar when already set but empty") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.registrator" -> "",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val autoTuner = buildDefaultDataprocAutoTuner(logEventsProps)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.kryo.registrator=com.nvidia.spark.rapids.GpuKryoRegistrator
          |--conf spark.kryoserializer.buffer.max=512m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.kryo.registrator' GpuKryoRegistrator must be appended to the existing value when using Kryo serialization.
          |- 'spark.kryoserializer.buffer.max' increasing the max buffer to prevent out-of-memory errors.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Test EMR sets shuffle manager properly and doesn't need Spark RAPIDS jar") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "32",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some("3.4.1-amzn-1"))
    val platform = PlatformFactory.createInstance(PlatformNames.EMR)

    // Configure cluster info using Platform's existing method
    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "212992MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 5,
      gpuCount = 4,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.executor.cores=16
          |--conf spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=13107m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=4915m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark341.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.driver.extraJavaOptions' was not set.
          |- 'spark.executor.extraJavaOptions' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- Set 'spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing driver JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- Set 'spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing executor JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("test AutoTuner sets Dataproc Spark performance enhancements") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val autoTuner = buildDefaultDataprocAutoTuner(logEventsProps)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("test AutoTuner skips Dataproc Spark performance enhancements if already set") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.dataproc.enhanced.execution.enabled" -> "false",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val autoTuner = buildDefaultDataprocAutoTuner(logEventsProps)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  Seq(true, false).foreach { hasGpuOOm =>
    test(s"test AutoTuner recommends default max partition bytes when not set " +
      s"[hasScanStagesWithGpuOom = $hasGpuOOm]") {
      // mock the properties loaded from eventLog
      val logEventsProps: mutable.Map[String, String] =
        mutable.LinkedHashMap[String, String](
          "spark.executor.cores" -> "16",
          "spark.executor.instances" -> "1",
          "spark.executor.memory" -> "80g",
          "spark.executor.resource.gpu.amount" -> "1",
          "spark.task.resource.gpu.amount" -> "0.001",
          "spark.rapids.memory.pinnedPool.size" -> "5g",
          "spark.rapids.sql.enabled" -> "true",
          "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
          "spark.rapids.sql.concurrentGpuTasks" -> "4")
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
        logEventsProps, Some(testSparkVersion),
        scanStagesWithGpuOom = if (hasGpuOOm) Set(1L) else Set.empty)
      val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

      // Configure cluster info using Platform's existing method
      val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "122880MiB")
      configureEventLogClusterInfoForTest(
        platform,
        numCores = 32,
        numWorkers = 4,
        gpuCount = 2,
        sparkProperties = sparkPropsWithMemory.toMap
      )

      val autoTuner = buildAutoTunerForTests(infoProvider, platform)
      val (properties, comments) = autoTuner.getRecommendedProperties()
      val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
      // scalastyle:off line.size.limit
      val expectedResults =
        s"""|
            |Spark Properties:
            |--conf spark.dataproc.enhanced.execution.enabled=false
            |--conf spark.dataproc.enhanced.optimizer.enabled=false
            |--conf spark.executor.memory=32g
            |--conf spark.executor.memoryOverhead=19660m
            |--conf spark.locality.wait=0
            |--conf spark.rapids.memory.pinnedPool.size=8g
            |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
            |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
            |--conf spark.rapids.sql.batchSizeBytes=2147483647b
            |--conf spark.rapids.sql.concurrentGpuTasks=3
            |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
            |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
            |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
            |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
            |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
            |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
            |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
            |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
            |--conf spark.sql.files.maxPartitionBytes=512m
            |
            |Comments:
            |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
            |- 'spark.dataproc.enhanced.execution.enabled' was not set.
            |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
            |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
            |- 'spark.executor.memoryOverhead' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
            |- 'spark.rapids.sql.batchSizeBytes' was not set.
            |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
            |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
            |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
            |- 'spark.shuffle.manager' was not set.
            |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
            |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
            |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
            |- 'spark.sql.files.maxPartitionBytes' was not set.
            |- ${classPathComments("rapids.jars.missing")}
            |- ${classPathComments("rapids.shuffle.jars")}
            |""".stripMargin
      // scalastyle:on line.size.limit
      compareOutput(expectedResults, autoTunerOutput)
    }
  }

  test("test AutoTuner reduces maxPartitionBytes when scan stages have GPU OOM failures") {
    val eventLog = s"$profilingLogDir/gpu_oom_eventlog.zstd"
    TrampolineUtil.withTempDir { tempDir =>
      val appArgs = new ProfileArgs(Array(
        "--csv",
        "--auto-tuner",
        "--output-directory",
        tempDir.getAbsolutePath,
        eventLog))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)

      // Assert that the maxPartitionBytes was 10gb in the source GPU event log
      val sparkPropertiesFile = getOutputFilePath(tempDir, "spark_properties.csv")
      val sparkProperties = FSUtils.readFileContentAsUTF8(sparkPropertiesFile)
      assert(sparkProperties.contains("\"spark.sql.files.maxPartitionBytes\",\"10gb\""))

      // Compare the auto-tuner output to the expected results and assert that
      // the maxPartitionBytes is reduced.
      val logFile = getOutputFilePath(tempDir, "profile.log")
      val profileLogContent = FSUtils.readFileContentAsUTF8(logFile)
      val actualResults = extractAutoTunerResults(profileLogContent)

      val testAppJarVer = "25.02.0"
      // scalastyle:off line.size.limit
      val expectedResults =
        s"""|
            |Spark Properties:
            |--conf spark.executor.instances=2
            |--conf spark.executor.memory=[FILL_IN_VALUE]
            |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
            |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
            |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
            |--conf spark.rapids.sql.batchSizeBytes=2147483647b
            |--conf spark.rapids.sql.concurrentGpuTasks=3
            |--conf spark.rapids.sql.enabled=true
            |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
            |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
            |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
            |--conf spark.sql.files.maxPartitionBytes=1851m
            |--conf spark.sql.shuffle.partitions=400
            |--conf spark.task.resource.gpu.amount=0.001
            |
            |Comments:
            |- 'spark.executor.instances' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
            |- 'spark.rapids.sql.batchSizeBytes' was not set.
            |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
            |- 'spark.rapids.sql.enabled' was not set.
            |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
            |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
            |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
            |- ${latestPluginJarComment(latestPluginJarUrl, testAppJarVer)}
            |- ${notEnoughMemCommentForKey("spark.executor.memory")}
            |- ${notEnoughMemCommentForKey("spark.rapids.memory.pinnedPool.size")}
            |- $shufflePartitionsCommentForSpilling
            |- ${classPathComments("rapids.shuffle.jars")}
            |- ${notEnoughMemComment(40140)}
            |- $missingGpuDiscoveryScriptComment
            |""".stripMargin.trim
      // scalastyle:on line.size.limit
      compareOutput(expectedResults, actualResults)
    }
  }

  // TODO: Revisit this test once optimal tuning for AQE for dataproc on g2 instances is determined
  //       See https://github.com/NVIDIA/spark-rapids-tools/issues/1682
  ignore(s"test AutoTuner recommends increasing shuffle partition when shuffle stages " +
    s"have OOM failures") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.sql.shuffle.partitions" -> "150", // AutoTuner should recommend increasing this
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion), gpuShuffleStagesWithContainerOom = Set(1L))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "122880MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=2
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.sql.shuffle.partitions=300
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.sql.shuffle.partitions' should be increased since task OOM occurred in shuffle stages.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // TODO: Revisit this test once optimal tuning for AQE for dataproc on g2 instances is determined
  //       See https://github.com/NVIDIA/spark-rapids-tools/issues/1682
  ignore(s"test AutoTuner recommends increasing shuffle partition and AQE initial partition num " +
    s"when shuffle stages have OOM failures") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.sql.shuffle.partitions" -> "50", // AutoTuner should recommend increasing this
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "4")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion), gpuShuffleStagesWithContainerOom = Set(1L),
      meanInput = 50000, meanShuffleRead = 80000)
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info using Platform's existing method
    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "122880MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 4,
      gpuCount = 2,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.instances=8
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=2
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=32m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=800
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.sql.shuffle.partitions=800
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.sql.shuffle.partitions' should be increased since task OOM occurred in shuffle stages.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // ============ Unit Tests for AutoTuner memory configurations ============
  // It verifies that the AutoTuner correctly handles memory configurations with and without units,
  // particularly for configurations whose default unit is not Byte (e.g. MB for memory overhead).
  //
  // Test Cases:
  // 1. Configs with explicit units (e.g. "17612m") are handled correctly and skipped if appropriate
  // 2. Configs without units (e.g. "17612") are handled correctly and skipped when appropriate
  // 3. Configs without units are properly updated with units when changes are needed

  /**
   * Helper method to test AutoTuner recommendations for memory configurations.
   *
   * @param testName      Name of the test case
   * @param confKey       The Spark configuration key to test
   * @param initialValue  Initial value of the configuration (with or without unit)
   * @param expectedValue Expected value after AutoTuner processing (None if no change expected)
   * @param extraProps    Additional properties needed for the test
   */
  private def testAutoTunerConf(
                                 testName: String,
                                 confKey: String,
                                 initialValue: String,
                                 expectedValue: Option[String],
                                 extraProps: Map[String, String] = Map.empty
                               ): Unit = {
    test(testName) {
      val baseProps = Map(
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1"
      )

      val testProps = Map(confKey -> initialValue) ++ extraProps
      val allProps = mutable.LinkedHashMap((baseProps ++ testProps).toSeq: _*)
      val autoTuner = buildDefaultDataprocAutoTuner(allProps)
      val (properties, comments) = autoTuner.getRecommendedProperties()
      val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

      def getAssertMessage(expectation: String): String = {
        s"""|=== Expected ===
            |$expectation
            |
            |=== Actual ===
            |$autoTunerOutput
            |""".stripMargin
      }

      expectedValue match {
        case Some(expected) =>
          val expectedLine = s"--conf $confKey=$expected"
          val hasExpectedLine = autoTunerOutput.contains(expectedLine)
          assert(hasExpectedLine,
            getAssertMessage(s"Expected to find '$expectedLine' in output but it was missing"))
        case None =>
          val prefix = s"--conf $confKey="
          val hasPrefix = autoTunerOutput.contains(prefix)
          assert(!hasPrefix,
            getAssertMessage(s"Expected NOT to find any conf with '$prefix' but found one"))
      }
    }
  }

  /**
   * Helper method to test executor memory overhead configurations.
   * Tests how the AutoTuner handles spark.executor.memoryOverhead with and without units.
   */
  private def testMemoryOverhead(
                                  testName: String,
                                  initialValue: String,
                                  expectedValue: Option[String]
                                ): Unit = {
    testAutoTunerConf(
      testName = testName,
      confKey = "spark.executor.memoryOverhead",
      initialValue = initialValue,
      expectedValue = expectedValue
    )
  }

  /**
   * Helper method to test Kryo serializer buffer configurations.
   * Tests how the AutoTuner handles spark.kryoserializer.buffer.max with and without units.
   */
  private def testKryoMaxBuffer(
                                 testName: String,
                                 initialValue: String,
                                 expectedValue: Option[String]
                               ): Unit = {
    testAutoTunerConf(
      testName = testName,
      confKey = "spark.kryoserializer.buffer.max",
      initialValue = initialValue,
      expectedValue = expectedValue,
      extraProps = Map("spark.serializer" -> "org.apache.spark.serializer.KryoSerializer")
    )
  }

  // Test cases for memory overhead configuration
  testMemoryOverhead(
    testName = "Test memory overhead with unit should be skipped when appropriate",
    initialValue = "19660m",
    expectedValue = None
  )

  testMemoryOverhead(
    testName = "Test memory overhead without unit should be skipped when appropriate",
    initialValue = "19660",
    expectedValue = None
  )

  testMemoryOverhead(
    testName = "Test memory overhead without unit should be updated and set",
    initialValue = "8712",
    expectedValue = Some("19660m")
  )

  // Test cases for Kryo buffer configuration
  testKryoMaxBuffer(
    testName = "Test kryo max buffer with unit should be skipped when appropriate",
    initialValue = "1024m",
    expectedValue = None
  )

  testKryoMaxBuffer(
    testName = "Test kryo max buffer without unit should be skipped when appropriate",
    initialValue = "1024",
    expectedValue = None
  )

  testKryoMaxBuffer(
    testName = "Test kryo max buffer without unit should be updated and set",
    initialValue = "128",
    expectedValue = Some("512m")
  )

  // Verifies that AutoTuner accounts for off-heap memory and recommends
  // memory configurations when sufficient memory is available for the executor.
  test("AutoTuner recommends memory configs when executor memory is sufficient after off-heap") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "30g",
        "spark.memory.offHeap.enabled" -> "true",
        "spark.memory.offHeap.size" -> "10g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)
    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=9420m
          |--conf spark.locality.wait=0
          |--conf spark.memory.offHeap.size=10240m
          |--conf spark.rapids.memory.pinnedPool.size=3g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Verifies that AutoTuner accounts for high off-heap memory and skips
  // recommendations for memory configurations when sufficient memory is
  // not available for the executor.
  test("AutoTuner warns and skips memory configs when executor memory is " +
    "not sufficient after off-heap") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "30g",
        "spark.memory.offHeap.enabled" -> "true",
        "spark.memory.offHeap.size" -> "20g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.instances" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4, // from buildGpuWorkerInfoFromInstanceType parameter
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=[FILL_IN_VALUE]
          |--conf spark.executor.memoryOverhead=[FILL_IN_VALUE]
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${notEnoughMemCommentForKey("spark.executor.memory")}
          |- ${notEnoughMemCommentForKey("spark.executor.memoryOverhead")}
          |- ${notEnoughMemCommentForKey("spark.rapids.memory.pinnedPool.size")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- ${notEnoughMemComment(75775)}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // IMPORTANT NOTE:
  // 1. This test suite is deprecated as it uses the legacy worker info properties format,
  //    which is overloaded to be used for both source and target cluster properties.
  // 2. All new Profiling AutoTuner test cases should be added to ProfilingAutoTunerSuiteV2
  //    instead of this suite.
  //
  // TODO:
  // Migrate all tests in this suite to use the new target cluster properties format.
  // https://github.com/NVIDIA/spark-rapids-tools/issues/1748
}
