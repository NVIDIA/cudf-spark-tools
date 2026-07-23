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

import java.nio.file.Paths

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.{DynamicAllocationInfo, GpuTypes, PlatformFactory, PlatformNames, ToolTestUtils}
import com.nvidia.spark.rapids.tool.profiling.Profiler
import com.nvidia.spark.rapids.tool.qualification.{QualificationArgs, QualificationMain}
import com.nvidia.spark.rapids.tool.tuning.config.{CategoryEnum, ConfTypeEnum, LevelEnum, TuningConfigEntry, TuningEntryDefinition}
import com.nvidia.spark.rapids.tool.views.CLUSTER_INFORMATION_LABEL
import com.nvidia.spark.rapids.tool.views.qualification.QualReportGenConfProvider
import org.scalatest.exceptions.TestFailedException
import org.scalatest.prop.TableDrivenPropertyChecks._
import org.scalatest.prop.TableFor3

import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.rapids.tool.{MatchingInstanceTypeNotFoundException, RecommendedClusterInfo}
import org.apache.spark.sql.rapids.tool.util.FSUtils

/**
 * Suite to test the Qualification Tool's AutoTuner
 */
class QualificationAutoTunerSuite extends BaseAutoTunerSuite {
  val qualLogDir: String = ToolTestUtils.getTestResourcePath("spark-events-qualification")
  val autoTunerHelper: AutoTunerHelper = QualificationAutoTunerHelper

  /**
   * Default Spark properties to be used when building the Qualification AutoTuner
   */
  private def defaultSparkProps: mutable.Map[String, String] = {
    mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "32",
      "spark.executor.instances" -> "1",
      "spark.executor.memory" -> "80g",
      "spark.executor.instances" -> "1"
    )
  }

  /**
   * Helper method to return an instance of the Qualification AutoTuner with default properties
   */
  private def buildDefaultAutoTuner(
      logEventsProps: mutable.Map[String, String] = defaultSparkProps): AutoTuner = {
    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "212992MiB")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      sparkPropsWithMemory, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.EMR)

    // Configure cluster info: 32 cores, 5 workers, 4 GPUs per worker = 20 total executors
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 32,
      execsPerNode = 4,
      numExecs = 20, // 5 workers * 4 GPUs per worker
      numExecutorNodes = 5,
      sparkProperties = sparkPropsWithMemory.toMap,
      systemProperties = Map.empty
    )

    buildAutoTunerForTests(infoProvider, platform)
  }

  /**
   * Helper method to check if the expected lines exist in the AutoTuner output.
   */
  private def assertExpectedLinesExist(
      expectedResults: Seq[String], autoTunerOutput: String): Unit = {
    val missingLines = expectedResults.filterNot(autoTunerOutput.contains)

    if (missingLines.nonEmpty) {
      val errorMessage =
        s"""|=== Missing Lines ===
            |${missingLines.mkString("\n")}
            |
            |=== Actual Output ===
            |$autoTunerOutput
            |""".stripMargin
      fail(errorMessage)
    }
  }

  test("test AutoTuner for Qualification sets batch size to 1GB") {
    val autoTuner = buildDefaultAutoTuner()
    val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps =
      QualificationAutoTunerRunner.filterByUpdatedPropsEnabled)
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    val expectedResults = Seq(
        "--conf spark.rapids.sql.batchSizeBytes=1g",
        "- 'spark.rapids.sql.batchSizeBytes' was not set."
    )
    assertExpectedLinesExist(expectedResults, autoTunerOutput)
  }

  test("test AutoTuner for Qualification should not change shuffle partitions") {
    // Set shuffle partitions to 100. The AutoTuner should recommend the same value
    // because currently shuffle.partitions is one of the limitedLogicRecommendations.
    // It will not be added to the recommendations because the value has not changed.
    val autoTuner = buildDefaultAutoTuner(
      defaultSparkProps ++ mutable.Map("spark.sql.shuffle.partitions" -> "100")
    )
    val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps =
      QualificationAutoTunerRunner.filterByUpdatedPropsEnabled)
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    val expectedResults = Seq(
      "--conf spark.sql.shuffle.partitions=100"
    )
    assertExpectedLinesExist(expectedResults, autoTunerOutput)
  }

  // scalastyle:off line.size.limit
  val testData: TableFor3[String, String, Seq[String]] = Table(
    ("testName", "workerMemory", "expectedResults"),
    ("less memory available for executors",
      "16g",
      Seq(
        "--conf spark.executor.memory=[FILL_IN_VALUE]",
        "--conf spark.executor.memoryOverhead=[FILL_IN_VALUE]",
        "--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]",
        s"- ${notEnoughMemCommentForKey("spark.executor.memory")}",
        s"- ${notEnoughMemCommentForKey("spark.executor.memoryOverhead")}",
        s"- ${notEnoughMemCommentForKey("spark.rapids.memory.pinnedPool.size")}",
        s"- ${notEnoughMemComment(40140)}"
      )),
    ("sufficient memory available for executors",
      "44g",
      Seq(
        "--conf spark.executor.memory=32g",
        "--conf spark.executor.memoryOverhead=14g",
        "--conf spark.rapids.memory.pinnedPool.size=5530m"
      ))
  )
  // scalastyle:on line.size.limit

  forAll(testData) { (testName: String, workerMemory: String, expectedResults: Seq[String]) =>
    test(s"test memory warnings for case: $testName") {
      val logEventsProps: mutable.Map[String, String] =
        mutable.LinkedHashMap[String, String](
          "spark.executor.cores" -> "8",
          "spark.executor.instances" -> "4",
          "spark.executor.memory" -> "8g",
          "spark.executor.memoryOverhead" -> "2g"
        )
      val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> workerMemory)
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
        logEventsProps, Some(testSparkVersion))
      val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
      // Configure cluster info: 16 cores, 2 workers, 2 GPUs per worker = 4 total executors
      platform.configureClusterInfoFromEventLog(
        coresPerExecutor = 16,
        execsPerNode = 2,
        numExecs = 4, // 2 workers * 2 GPUs per worker
        numExecutorNodes = 2,
        sparkProperties = sparkPropsWithMemory.toMap,
        systemProperties = Map.empty
      )
      val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn))
      val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps =
        QualificationAutoTunerRunner.filterByUpdatedPropsEnabled)
      val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
      assertExpectedLinesExist(expectedResults, autoTunerOutput)
    }
  }

  /**
   * Test to validate the cluster shape recommendation with enforced spark properties.
   * This tests that if the user has enforced `spark.executor.instances`, this will
   * affect the recommended cluster shape.
   *
   * Target Cluster YAML file:
   * {{{
   * driverInfo:
   *  instanceType: n1-standard-8
   * workerInfo:
   *  instanceType: g2-standard-8
   * sparkProperties:
   *  enforced:
   *    spark.executor.cores: 8
   *    spark.executor.instances: 4
   *    spark.executor.memory: 12g
   * }}}
   */
  test(s"test valid cluster shape recommendation with enforced spark properties on dataproc " +
    s"affecting the cluster shape") {
    val testEventLog = s"$qualLogDir/nds_q72_dataproc_2_2.zstd"
    val testEnforcedSparkProperties = Map(
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "4",
      "spark.rapids.sql.batchSizeBytes" -> "3g"
    )
    val expectedClusterInfo = RecommendedClusterInfo(
      vendor = PlatformNames.DATAPROC,
      coresPerExecutor = 8,
      numWorkerNodes = 4,
      numGpusPerNode = 1,
      numExecutors = 4,
      gpuDevice = "nvidia-l4",
      dynamicAllocationEnabled = false,
      dynamicAllocationMaxExecutors = "N/A",
      dynamicAllocationMinExecutors = "N/A",
      dynamicAllocationInitialExecutors = "N/A",
      driverNodeType = Some("n1-standard-8"),
      workerNodeType = Some("g2-standard-8")
    )
    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        driverNodeInstanceType = expectedClusterInfo.driverNodeType,
        workerNodeInstanceType = expectedClusterInfo.workerNodeType,
        enforcedSparkProperties = testEnforcedSparkProperties)

      val appArgs = new QualificationArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        "--auto-tuner",
        testEventLog
        ))

      val result = QualificationMain.mainInternal(appArgs)
      assert(!result.isFailed)
      val appId = result.appSummaries.headOption.map(_.appId)
        .getOrElse(throw new TestFailedException("No appId found in the result", 0))

      // 1. Verify the recommended cluster info
      val clusterInfoFileName = s"${CLUSTER_INFORMATION_LABEL.replace(" ", "_").toLowerCase}.json"
      val actualClusterInfoFile = Paths.get(
        QualReportGenConfProvider.getPerAppReportPath(tempDir.getAbsolutePath),
        appId, clusterInfoFileName
      ).toFile
      assertRecommendedClusterInfo(actualClusterInfoFile, expectedClusterInfo)

      // 2. Verify the enforced spark properties
      val tuningResultPath = Paths.get(
        QualReportGenConfProvider.getTuningReportPath(tempDir.getAbsolutePath),
        s"$appId.log"
      ).toString
      val actualTuningResults = FSUtils.readFileContentAsUTF8(tuningResultPath)

      // scalastyle:off line.size.limit
      val expectedResults =
        s"""|
            |### Recommended SPARK Configuration on GPU Cluster for App: $appId ###
            |
            |Spark Properties:
            |--conf spark.dataproc.enhanced.execution.enabled=true
            |--conf spark.dataproc.enhanced.optimizer.enabled=true
            |--conf spark.executor.cores=8
            |--conf spark.executor.instances=4
            |--conf spark.executor.memory=16g
            |--conf spark.executor.memoryOverhead=9830m
            |--conf spark.executor.resource.gpu.amount=1
            |--conf spark.locality.wait=0
            |--conf spark.plugins=com.nvidia.spark.SQLPlugin
            |--conf spark.rapids.memory.pinnedPool.size=4g
            |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
            |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
            |--conf spark.rapids.sql.batchSizeBytes=3g
            |--conf spark.rapids.sql.concurrentGpuTasks=3
            |--conf spark.rapids.sql.enabled=true
            |--conf spark.rapids.sql.multiThreadedRead.numThreads=40
            |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
            |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=128
            |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
            |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
            |--conf spark.sql.adaptive.enabled=true
            |--conf spark.sql.files.maxPartitionBytes=1644m
            |--conf spark.sql.shuffle.partitions=128
            |--conf spark.task.resource.gpu.amount=0.001
            |
            |Comments:
            |- ${getEnforcedPropertyComment("spark.executor.cores")}
            |- ${getEnforcedPropertyComment("spark.executor.instances")}
            |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
            |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
            |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
            |- 'spark.rapids.memory.pinnedPool.size' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
            |- ${getEnforcedPropertyComment("spark.rapids.sql.batchSizeBytes")}
            |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
            |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
            |- $shuffleManagerCommentForQualification
            |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
            |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
            |- 'spark.sql.files.maxPartitionBytes' was not set.
            |- 'spark.task.resource.gpu.amount' was not set.
            |- ${classPathComments("rapids.jars.missing")}
            |- ${classPathComments("rapids.shuffle.jars")}
            |- $additionalSparkPluginsComment
            |""".stripMargin.trim
      // scalastyle:on line.size.limit
      compareOutput(expectedResults, actualTuningResults)
    }
  }

  // This test validates that user-provided tuning configurations specific to Qualification
  // are honored by the AutoTuner.
  // AutoTuner is expected to:
  // - Recommend `spark.executor.memory` to a value:
  //     1.2g/core * 16cores = 19648m
  // - Recommend `spark.rapids.sql.concurrentGpuTasks` to a value:
  //     max(CONC_GPU_TASKS (8), gpuMemory (24g) / GPU_MEM_PER_TASK (4g) = 6
  test("AutoTuner honours user provided tuning configurations specific to Qualification") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "4",
        "spark.executor.memory" -> "8g",
        "spark.executor.memoryOverhead" -> "2g"
      )
    // 2. Mock the user-provided tuning configurations. Equivalent YAML snippet:
    // tuningConfigs:
    //   default:
    //   - name: GPU_MEM_PER_TASK
    //     default: 4g
    //   - name: CONC_GPU_TASKS
    //     max: 8
    //   qualification:
    //   - name: HEAP_PER_CORE
    //     default: 1.2g
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "GPU_MEM_PER_TASK", default = "4g"),
      TuningConfigEntry(name = "CONC_GPU_TASKS", max = "8")
    )
    val qualificationTuningConfigEntries = List(
      TuningConfigEntry(name = "HEAP_PER_CORE", default = "1.2g")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries, qualification = qualificationTuningConfigEntries)
    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "32g")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

    // Configure cluster info: 8 cores, 2 workers, 2 GPUs per worker = 4 total executors
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 2,
      numExecs = 4, // 2 workers * 2 GPUs per worker
      numExecutorNodes = 2,
      sparkProperties = sparkPropsWithMemory.toMap,
      systemProperties = Map.empty
    )

    val autoTuner =
      buildAutoTunerForTests(infoProvider, platform, Some(Yarn), Some(userProvidedTuningConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=16
          |--conf spark.executor.instances=2
          |--conf spark.executor.memory=19648m
          |--conf spark.executor.memoryOverhead=15168m
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=6602m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=6
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $missingGpuDiscoveryScriptComment
          |- $additionalSparkPluginsComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test to validate that Bootstrap sets the appropriate GPU resource properties (i.e. amount,
  // discovery script and vendor) based on the Spark master type.
  // scalastyle:off line.size.limit
  val gpuResourcePropertiesTestData: TableFor3[String, SparkMaster, Seq[String]] = Table(
    ("testName", "sparkMaster", "expectedResults"),
    ("Standalone",
      Standalone,
      Seq(
        "--conf spark.executor.resource.gpu.amount=1",
        "- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.",
        s"- $missingGpuDiscoveryScriptComment"
      )),
    ("Yarn",
      Yarn,
      Seq(
        "--conf spark.executor.resource.gpu.amount=1",
        "- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.",
        s"- $missingGpuDiscoveryScriptComment"
      )),
    ("Kubernetes",
      Kubernetes,
      Seq(
        "--conf spark.executor.resource.gpu.amount=1",
        "--conf spark.executor.resource.gpu.vendor=nvidia.com",
        "- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.",
        "- 'spark.executor.resource.gpu.vendor' was not set.",
        s"- $missingGpuDiscoveryScriptComment"
      ))
  )
  // scalastyle:on line.size.limit

  forAll(gpuResourcePropertiesTestData) {
    (testName: String, sparkMaster: SparkMaster, expectedResults: Seq[String]) =>
      test(s"test AutoTuner for Qualification sets GPU resource properties for $testName") {
        val sparkPropsWithMemory = defaultSparkProps + ("spark.executor.memory" -> "122880MiB")
        val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
          defaultSparkProps, Some(testSparkVersion))
        val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

        // Configure cluster info: 32 cores, 4 workers, 2 GPUs per worker = 8 total executors
        platform.configureClusterInfoFromEventLog(
          coresPerExecutor = 32,
          execsPerNode = 2,
          numExecs = 8, // 4 workers * 2 GPUs per worker
          numExecutorNodes = 4,
          sparkProperties = sparkPropsWithMemory.toMap,
          systemProperties = Map.empty
        )

        val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(sparkMaster))
        val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps =
          QualificationAutoTunerRunner.filterByUpdatedPropsEnabled)
        val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
        assertExpectedLinesExist(expectedResults, autoTunerOutput)
      }
  }

  // This test ensures that AutoTuner honours enforced values for the GPU discovery script
  // and does not include a missing discovery script comment when spark master
  // is YARN
  test("test AutoTuner honours enforced gpu discovery script and" +
    " skips the missing comment when spark master is YARN") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "32g",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    // Define 'spark.executor.resource.gpu.discoveryScript' as an enforced property
    val enforcedSparkProperties = Map(
      "spark.executor.resource.gpu.discoveryScript" -> "/opt/sparkPlugin/gpuDiscoveryScript.sh"
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = enforcedSparkProperties
    )
    val sparkPropsWithMemory = defaultSparkProps + ("spark.executor.memory" -> "122880MiB")
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 32,
      execsPerNode = 2,
      numExecs = 8,
      numExecutorNodes = 4,
      sparkProperties = sparkPropsWithMemory.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform,
      sparkMaster = Some(Yarn))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=16g
          |--conf spark.executor.memoryOverhead=109772m
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.executor.resource.gpu.discoveryScript=/opt/sparkPlugin/gpuDiscoveryScript.sh
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.executor.resource.gpu.discoveryScript' was user-enforced in the target cluster properties.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $additionalSparkPluginsComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  /**
   * Test to validate that enforced properties from target cluster info are included in bootstrap.
   * This tests that properties specified in sparkProperties.enforced section appear in both
   * the .log file and the -bootstrap.conf file (regardless of whether they are in tuning table).
   */
  test("test enforced properties are included in bootstrap config") {
    val testEventLog = s"$qualLogDir/nds_q72_dataproc_2_2.zstd"
    // Include both tuning table properties and non-tuning table properties
    val testEnforcedSparkProperties = Map(
      "spark.executor.cores" -> "8",              // In tuning table
      "spark.sql.shuffle.partitions" -> "400",    // In tuning table
      "spark.custom.property" -> "customValue",   // Not in tuning table
      "spark.app.name" -> "TestApp"               // Not in tuning table
    )

    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        driverNodeInstanceType = Some("n1-standard-8"),
        workerNodeInstanceType = Some("g2-standard-8"),
        enforcedSparkProperties = testEnforcedSparkProperties)

      val appArgs = new QualificationArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        "--auto-tuner",
        testEventLog
      ))

      val result = QualificationMain.mainInternal(appArgs)
      assert(!result.isFailed)
      val appId = result.appSummaries.headOption.map(_.appId)
        .getOrElse(throw new TestFailedException("No appId found in the result", 0))

      // 1. Verify that enforced properties appear in the main tuning log
      val tuningResultPath = Paths.get(
        QualReportGenConfProvider.getTuningReportPath(tempDir.getAbsolutePath),
        s"$appId.log"
      ).toString
      val actualTuningResults = FSUtils.readFileContentAsUTF8(tuningResultPath)

      testEnforcedSparkProperties.keys.foreach { propertyName =>
        assert(actualTuningResults.contains(s"--conf $propertyName="),
          s"Property $propertyName should appear in tuning log")
        assert(actualTuningResults.contains(getEnforcedPropertyComment(propertyName)),
          s"Enforced property comment for $propertyName should appear in tuning log")
      }

      // 2. Verify that ALL enforced properties also appear in the bootstrap config
      val bootstrapConfigPath = Paths.get(
        QualReportGenConfProvider.getTuningReportPath(tempDir.getAbsolutePath),
        s"$appId-bootstrap.conf"
      ).toString
      val bootstrapConfigContent = FSUtils.readFileContentAsUTF8(bootstrapConfigPath)

      testEnforcedSparkProperties.foreach { case (propertyName, propertyValue) =>
        assert(bootstrapConfigContent.contains(s"--conf $propertyName=$propertyValue"),
          s"Enforced property $propertyName=$propertyValue should appear in bootstrap config")
      }
    }
  }

  /**
   * Test to validate onPrem platform with offHeapLimit enabled.
   * This tests the new memory calculation logic with NON_EXECUTOR_MEM and offHeapLimit features.
   */
  test("test onPrem platform with offHeapLimit enabled") {
    // Log events properties
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "4",
      "spark.executor.memory" -> "6144M",
      "spark.executor.instances" -> "20"
    )

    // Enforced Spark properties
    val enforcedSparkProps = Map(
      "spark.executor.cores" -> "20",
      "spark.shuffle.manager" -> "org.apache.spark.shuffle.celeborn.SparkShuffleManager",
      "spark.rapids.sql.multiThreadedRead.numThreads" -> "250",
      "spark.vcore.boost.ratio" -> "4",
      "spark.memory.offHeap.enabled" -> "true",
      "spark.memory.offHeap.size" -> "45g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
      "spark.rapids.memory.host.offHeapLimit.enabled" -> "true",
      "spark.rapids.memory.host.offHeapLimit.size" -> "80g",
      "spark.sql.adaptive.enabled" -> "true"
    )

      // Build target cluster info with worker configuration and enforced properties
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(20),
        memoryGB = Some(120L), // 120g = 120GB
        gpuCount = Some(1),
        gpuMemory = Some("48g"),
        gpuDevice = Some("l20"),
        enforcedSparkProperties = enforcedSparkProps
      )

      val infoProvider = getMockInfoProvider(
        maxInput = 0.0,
        spilledMetrics = Seq(0),
        jvmGCFractions = Seq(0.0),
        propsFromLog = logEventsProps,
        sparkVersion = Some(testSparkVersion)
      )

    // tuningConfigs:
    //   default:
    //   - name: HEAP_PER_CORE
    //     default: 1g
    //   - name: CONC_GPU_TASKS
    //     max: 2
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "HEAP_PER_CORE", default = "1g"),
      TuningConfigEntry(name = "CONC_GPU_TASKS", max = "2"),
      TuningConfigEntry(name = "NON_EXECUTOR_MEM", default = "5g")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)

    // Create platform with target cluster info
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(
      platform = platform,
      numCores = 20,
      numWorkers = 4,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    // Build AutoTuner
    val autoTuner = buildAutoTunerForTests(infoProvider, platform,
      sparkMaster = Some(Kubernetes), userProvidedTuningConfigs = Some(userProvidedTuningConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // Per memory tune logic when offHeapLimit is enabled:
    // totalMemoryMinusReserved = 120(enforced) - 5 = 115g
    // sparkOffHeapMemMB = 45g(enforced)
    // overhead = 115g - 20g - 45g = 50g
    // pinned = min( (45(sparkOffHeapMemMB) + 50(overhead) / 4), 20 * 2(OFFHEAP_PER_CORE))
    //        = 24320m
    // Expected results for onPrem with offHeapLimit enabled
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=20
          |--conf spark.executor.instances=4
          |--conf spark.executor.memory=20g
          |--conf spark.executor.memoryOverhead=50g
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.executor.resource.gpu.vendor=nvidia.com
          |--conf spark.locality.wait=0
          |--conf spark.memory.offHeap.enabled=true
          |--conf spark.memory.offHeap.size=45g
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.host.offHeapLimit.enabled=true
          |--conf spark.rapids.memory.host.offHeapLimit.size=80g
          |--conf spark.rapids.memory.pinnedPool.size=40g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=30
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=30
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=2
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=250
          |--conf spark.shuffle.manager=org.apache.spark.shuffle.celeborn.SparkShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |--conf spark.vcore.boost.ratio=4
          |
          |Comments:
          |- ${getEnforcedPropertyComment("spark.executor.cores")}
          |- 'spark.executor.memoryOverhead' was not set.
          |- ${getEnforcedPropertyComment("spark.executor.resource.gpu.amount")}
          |- 'spark.executor.resource.gpu.vendor' was not set.
          |- ${getEnforcedPropertyComment("spark.memory.offHeap.enabled")}
          |- ${getEnforcedPropertyComment("spark.memory.offHeap.size")}
          |- ${getEnforcedPropertyComment("spark.plugins")}
          |- ${getEnforcedPropertyComment("spark.rapids.memory.host.offHeapLimit.enabled")}
          |- ${getEnforcedPropertyComment("spark.rapids.memory.host.offHeapLimit.size")}
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- ${getEnforcedPropertyComment("spark.rapids.sql.multiThreadedRead.numThreads")}
          |- ${getEnforcedPropertyComment("spark.shuffle.manager")}
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${getEnforcedPropertyComment("spark.sql.adaptive.enabled")}
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${getEnforcedPropertyComment("spark.vcore.boost.ratio")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $missingGpuDiscoveryScriptComment
          |""".stripMargin
    // Verify expected results match output
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test verifies that AutoTuner recommends the correct value for
  // "spark.plugins" property.
  test("test 'spark.plugins' is recommended correctly") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "32g",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "122880MiB")
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 32,
      execsPerNode = 2,
      numExecs = 8,
      numExecutorNodes = 4,
      sparkProperties = sparkPropsWithMemory.toMap,
      systemProperties = Map.empty
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=16g
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $additionalSparkPluginsComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test verifies that Qualification Bootstrap ignores existing
  // "spark.plugins" property and RAPIDS plugin is added.
  test("test existing 'spark.plugins' are ignored and RAPIDS plugin is added") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "32g",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.plugins" -> "com.existing.plugin1,com.existing.plugin2")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 32,
      execsPerNode = 2,
      numExecs = 8,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=16g
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=7373m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should include the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $additionalSparkPluginsComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test verifies that AutoTuner honours enforced values of spark.plugins
  test("test enforced values of 'spark.plugins' are honoured by AutoTuner") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "32g",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))

    // Define 'spark.plugins' as an enforced property
    val enforcedSparkProperties = Map(
      "spark.plugins" -> "com.existing.plugin1,com.existing.plugin2"
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = enforcedSparkProperties
    )

    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 32,
      execsPerNode = 2,
      numExecs = 8,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=16g
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.existing.plugin1,com.existing.plugin2
          |--conf spark.rapids.memory.pinnedPool.size=7373m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- ${getEnforcedPropertyComment("spark.plugins")}
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("test multithread read core multiplier category config is specified" +
    " in the target cluster and defined in tuning definitions") {
    // Mock properties from event log
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "4",
      "spark.executor.memory" -> "16g"
    )

    // Define core multiplier property in enforced section
    val enforcedSparkProperties = Map(
      "com.custom.spark.coreMultiplier" -> "2.0"
    )

    // Create tuning definitions for the core multiplier property in the target cluster
    import scala.jdk.CollectionConverters._
    val coreMultiplierTuningDef = TuningEntryDefinition(
      label = "com.custom.spark.coreMultiplier",
      description = "Core multiplier property",
      confType = ConfTypeEnum.Double,
      level = LevelEnum.Cluster,
      category = CategoryEnum.MultiThreadReadCoreMultiplier
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(32),
      memoryGB = Some(128L),
      gpuCount = Some(1),
      gpuMemory = Some("24g"),
      gpuDevice = Some("a100"),
      enforcedSparkProperties = enforcedSparkProperties,
      tuningDefinitions = List(coreMultiplierTuningDef).asJava
    )

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))

    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 1,
      numExecs = 4,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // With multiplier of 2.0 -> From target cluster
    // Expected results should reflect this multiplier in calculations:
    // - multiThreadedRead.numThreads = 32 * 2.0 = 64
    val expectedResults = Seq(
      "--conf com.custom.spark.coreMultiplier=2.0",
      "--conf spark.rapids.sql.multiThreadedRead.numThreads=64",
      "- 'com.custom.spark.coreMultiplier' was user-enforced in the target cluster properties."
    )

    assertExpectedLinesExist(expectedResults, autoTunerOutput)
  }

  test("test multithread read core multiplier category config is specified" +
    " in the event log and defined in tuning definitions") {
    // Mock properties from event log including the core multiplier property
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "4",
      "spark.executor.instances" -> "8",
      "spark.executor.memory" -> "8g",
      "com.custom.spark.coreMultiplier" -> "3.0"
    )

    // Create tuning definitions for the core multiplier property
    import scala.jdk.CollectionConverters._
    val coreMultiplierTuningDef = TuningEntryDefinition(
      label = "com.custom.spark.coreMultiplier",
      description = "Core multiplier property",
      confType = ConfTypeEnum.Double,
      level = LevelEnum.Cluster,
      category = CategoryEnum.MultiThreadReadCoreMultiplier
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(32),
      memoryGB = Some(128L),
      gpuCount = Some(1),
      gpuMemory = Some("24g"),
      gpuDevice = Some("a100"),
      tuningDefinitions = List(coreMultiplierTuningDef).asJava
    )

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))

    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 4,
      execsPerNode = 1,
      numExecs = 8,
      numExecutorNodes = 8,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // With multiplier of 3.0 -> From event log
    // Expected results should reflect this multiplier in calculations:
    // - multiThreadedRead.numThreads = 32 * 3.0 = 96
    val expectedResults = Seq(
      "--conf spark.rapids.sql.multiThreadedRead.numThreads=96"
    )

    assertExpectedLinesExist(expectedResults, autoTunerOutput)

    // Verify the multiplier property is not in recommendations since it's unchanged
    assert(!autoTunerOutput.contains("--conf com.custom.spark.coreMultiplier=3.0"),
      "Core multiplier property should not appear in recommendations when unchanged from event log")
  }

  test("test multithread read core multiplier category config is specified" +
    " in the event log and but not defined in tuning definitions") {
    // Mock properties from event log including the core multiplier property
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "6",
      "spark.executor.instances" -> "4",
      "spark.executor.memory" -> "12g",
      "com.custom.spark.coreMultiplier" -> "2.5"
    )

    // Create target cluster info WITHOUT tuning definitions for the multiplier property
    // This means the multiplier should be ignored
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(32),
      memoryGB = Some(128L),
      gpuCount = Some(1),
      gpuMemory = Some("24g"),
      gpuDevice = Some("a100")
    )

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))

    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 6,
      execsPerNode = 1,
      numExecs = 4,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // With NO multiplier specified as property -> Use default multiplier from tuning configs
    // Expected results should reflect normal core calculations
    // - multiThreadedRead.numThreads = 32 * 2 = 64
    val expectedResults = Seq(
      "--conf spark.rapids.sql.multiThreadedRead.numThreads=64"
    )

    assertExpectedLinesExist(expectedResults, autoTunerOutput)
  }

  test("test multithread read core multiplier config is specified in the tuning configs") {
    // Mock properties from event log
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "6",
      "spark.executor.instances" -> "4",
      "spark.executor.memory" -> "12g"
    )

    // Create target cluster info WITHOUT tuning definitions for the multiplier property
    // This means the multiplier should be ignored
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(32),
      memoryGB = Some(128L),
      gpuCount = Some(1),
      gpuMemory = Some("24g"),
      gpuDevice = Some("a100")
    )

    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "MULTITHREAD_READ_CORE_MULTIPLIER", default = "5"),
      TuningConfigEntry(name = "MULTITHREAD_READ_NUM_THREADS", max = "100")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))

    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 6,
      execsPerNode = 1,
      numExecs = 4,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform,
      userProvidedTuningConfigs = Some(userProvidedTuningConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // Use the multiplier from the user-provided tuning configs
    // Expected results should reflect normal core calculations
    // - multiThreadedRead.numThreads = min(100, 32 * 5) = 100
    val expectedResults = Seq(
      "--conf spark.rapids.sql.multiThreadedRead.numThreads=100"
    )

    assertExpectedLinesExist(expectedResults, autoTunerOutput)
  }

  // This test verifies that an error is raised if there are overlapping keys
  // between exclude, preserve and enforced properties. In this case,
  // 'spark.sql.files.maxPartitionBytes' is in both exclude and preserve lists
  // and 'spark.sql.shuffle.partitions' is in both preserve and enforced lists
  test("test exclude, preserve and enforced properties in target cluster with " +
    "overlapping keys raise error") {
    val excludeProperties = List(
      "spark.rapids.sql.concurrentGpuTasks",
      "spark.rapids.shuffle.multiThreaded.writer.threads",
      "spark.rapids.shuffle.multiThreaded.reader.threads",
      "spark.sql.files.maxPartitionBytes"
    )
    val preserveProperties = List(
      "spark.sql.files.maxPartitionBytes",
      "spark.sql.shuffle.partitions"
    )
    val enforcedSparkProperties = Map(
      "spark.sql.shuffle.partitions" -> "800"
    )
    val exception = intercept[IllegalArgumentException] {
      ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(16),
        memoryGB = Some(64L),
        gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.A100),
        preserveSparkProperties = preserveProperties,
        enforcedSparkProperties = enforcedSparkProperties,
        excludeSparkProperties = excludeProperties)
    }

    // Verify the exception message contains the expected overlapping keys
    val expectedOverlappingKeys = Set(
      "spark.sql.files.maxPartitionBytes",  // in both exclude and preserve
      "spark.sql.shuffle.partitions"       // in both preserve and enforced
    )

    expectedOverlappingKeys.foreach { key =>
      assert(exception.getMessage.contains(key),
        s"Exception message should contain overlapping key: $key. Message: ${exception.getMessage}")
    }
  }

  test("test preserve properties not found in source") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "40g"
        // Note: spark.sql.shuffle.partitions is NOT in source properties
      )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))

    // Try to preserve a property that doesn't exist in source
    val preserveProperties = List(
      "spark.sql.shuffle.partitions" // This doesn't exist in source
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16),
      memoryGB = Some(64L),
      gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.A100),
      preserveSparkProperties = preserveProperties
    )

    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 16,
      execsPerNode = 1,
      numExecs = 2,
      numExecutorNodes = 2,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=32g
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=4
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.sql.shuffle.partitions=200
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- ${getPreservedPropertyNotFoundComment("spark.sql.shuffle.partitions")}
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $additionalSparkPluginsComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("test e2e exclude, preserve and enforced properties in target cluster") {
    val testEventLog = s"$qualLogDir/nds_q72_dataproc_2_2.zstd"

    val excludeProperties = List(
      // Exclude a property that is set in event log
      "spark.master",
      // Exclude a property that is recommended by AutoTuner
      "spark.rapids.sql.concurrentGpuTasks"
    )
    val preserveProperties = List(
      // Preserve a property that is not present in event log
      "spark.task.resource.gpu.amount",
      // Preserve a property that is present and recommended by AutoTuner
      "spark.executor.memory",
      // Preserve a property that is present but not recommended by AutoTuner
      "spark.dataproc.engine"
    )
    val enforcedSparkProperties = Map(
      "spark.sql.shuffle.partitions" -> "800"
    )

    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        preserveSparkProperties = preserveProperties,
        enforcedSparkProperties = enforcedSparkProperties,
        excludeSparkProperties = excludeProperties)

      val appArgs = new QualificationArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        "--auto-tuner",
        testEventLog
      ))

      val result = QualificationMain.mainInternal(appArgs)
      assert(!result.isFailed)
      val appId = result.appSummaries.headOption.map(_.appId)
        .getOrElse(throw new TestFailedException("No appId found in the result", 0))

      // 1. Verify that enforced properties appear in the combined tuning log
      val combinedResultPath = Paths.get(
        QualReportGenConfProvider.getTuningReportPath(tempDir.getAbsolutePath),
        s"$appId.conf"
      ).toString
      val actualCombinedResults = FSUtils.readFileContentAsUTF8(combinedResultPath)

      enforcedSparkProperties.keys.foreach { propertyName =>
        assert(actualCombinedResults.contains(s"--conf $propertyName="),
          s"Property $propertyName should appear in tuning log")
      }

      excludeProperties.foreach { propertyName =>
        assert(!actualCombinedResults.contains(s"--conf $propertyName="),
          s"Excluded Property $propertyName should NOT appear in tuning log")
      }

      // 2. Verify the tuning results contain expected properties and comments
      val tuningResultPath = Paths.get(
        QualReportGenConfProvider.getTuningReportPath(tempDir.getAbsolutePath),
        s"$appId.log"
      ).toString
      val actualTuningResults = FSUtils.readFileContentAsUTF8(tuningResultPath)

      // scalastyle:off line.size.limit
      val expectedResults =
        s"""|
            |### Recommended SPARK Configuration on GPU Cluster for App: $appId ###
            |
            |Spark Properties:
            |--conf spark.dataproc.engine=default
            |--conf spark.dataproc.enhanced.execution.enabled=true
            |--conf spark.dataproc.enhanced.optimizer.enabled=true
            |--conf spark.executor.cores=16
            |--conf spark.executor.instances=8
            |--conf spark.executor.memory=40g
            |--conf spark.executor.memoryOverhead=11468m
            |--conf spark.executor.resource.gpu.amount=1
            |--conf spark.locality.wait=0
            |--conf spark.plugins=com.nvidia.spark.SQLPlugin
            |--conf spark.rapids.memory.pinnedPool.size=3686m
            |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
            |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
            |--conf spark.rapids.sql.batchSizeBytes=1g
            |--conf spark.rapids.sql.enabled=true
            |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
            |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
            |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
            |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
            |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=800
            |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
            |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
            |--conf spark.sql.adaptive.enabled=true
            |--conf spark.sql.files.maxPartitionBytes=1644m
            |--conf spark.sql.shuffle.partitions=800
            |--conf spark.task.resource.gpu.amount=0.001
            |
            |Comments:
            |- ${getPreservedPropertyComment("spark.dataproc.engine")}
            |- ${getPreservedPropertyComment("spark.executor.memory")}
            |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
            |- ${getExcludedPropertyComment("spark.master")}
            |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
            |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
            |- 'spark.rapids.memory.pinnedPool.size' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
            |- 'spark.rapids.sql.batchSizeBytes' was not set.
            |- ${getExcludedPropertyComment("spark.rapids.sql.concurrentGpuTasks")}
            |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
            |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
            |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
            |- $shuffleManagerCommentForQualification
            |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
            |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
            |- 'spark.sql.files.maxPartitionBytes' was not set.
            |- ${getEnforcedPropertyComment("spark.sql.shuffle.partitions")}
            |- 'spark.task.resource.gpu.amount' was not set.
            |- ${getPreservedPropertyNotFoundComment("spark.task.resource.gpu.amount")}
            |- ${classPathComments("rapids.jars.missing")}
            |- ${classPathComments("rapids.shuffle.jars")}
            |- $additionalSparkPluginsComment
            |""".stripMargin.trim
      // scalastyle:on line.size.limit
      compareOutput(expectedResults, actualTuningResults)
    }
  }

  /**
   * Test that AutoTuner recommends increasing 'spark.sql.shuffle.partitions' when shuffle stage
   * spilling is detected. For example, if the original value is 200 and spilling occurs, the
   * recommended value should be higher (e.g., 400).
   */
  test("test AutoTuner increases shuffle partitions recommendation when shuffle stage" +
    " spilling is detected") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "32g",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g")
    val infoProvider = getMockInfoProvider(3.7449728E7, Seq(1000L, 1000L), Seq(0.4, 0.4),
      logEventsProps, Some(testSparkVersion), shuffleStagesWithPosSpilling = Set(1))

    // Define 'spark.plugins' as an enforced property
    val enforcedSparkProperties = Map(
      "spark.plugins" -> "com.existing.plugin1,com.existing.plugin2"
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = enforcedSparkProperties
    )

    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 32,
      execsPerNode = 2,
      numExecs = 8,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=16g
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.existing.plugin1,com.existing.plugin2
          |--conf spark.rapids.memory.pinnedPool.size=7373m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=3669m
          |--conf spark.sql.shuffle.partitions=400
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- ${getEnforcedPropertyComment("spark.plugins")}
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- Average JVM GC time is very high. Other Garbage Collectors can be used for better performance.
          |- ${classPathComments("rapids.jars.missing")}
          |- $shufflePartitionsCommentForSpilling
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test AutoTuner recommends adjusting dynamic allocation properties when dynamic allocation
  // is enabled based on the formula:  max(1, floor(CPU_value × CPU_cores / GPU_cores)).
  test("test AutoTuner recommends dynamic allocation properties when enabled") {
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "20",
      "spark.executor.memory" -> "16g",
      "spark.dynamicAllocation.enabled" -> "true",
      "spark.dynamicAllocation.initialExecutors" -> "12",
      "spark.dynamicAllocation.minExecutors" -> "12",
      "spark.dynamicAllocation.maxExecutors" -> "30"
    )

    val expectedAdjustedProperties = List(
      // 'spark.dynamicAllocation.initialExecutors' is excluded from this list since in this
      // case its final value is not determined by the formula but by the recommended value
      // from 'executor.instances'.
      "spark.dynamicAllocation.minExecutors",
      "spark.dynamicAllocation.maxExecutors"
    )

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.EMR)
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 4,
      numExecs = 20,
      numExecutorNodes = 5,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.dynamicAllocation.initialExecutors=10
          |--conf spark.dynamicAllocation.maxExecutors=15
          |--conf spark.dynamicAllocation.minExecutors=6
          |--conf spark.executor.cores=16
          |--conf spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.executor.instances=10
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=13107m
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=4915m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
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
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- Set 'spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing driver JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- Set 'spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing executor JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $additionalSparkPluginsComment
          |- ${commentForDynamicAllocationAdjustment(expectedAdjustedProperties, 8, 16)}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("test AutoTuner does not recommend dynamic allocation properties when disabled") {
    val logEventsProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "4",
      "spark.executor.memory" -> "16g",
      "spark.dynamicAllocation.enabled" -> "false",
      "spark.dynamicAllocation.initialExecutors" -> "10",
      "spark.dynamicAllocation.minExecutors" -> "5",
      "spark.dynamicAllocation.maxExecutors" -> "20"
    )

    val infoProviderDisabled = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platformDisabled = PlatformFactory.createInstance(PlatformNames.ONPREM)
    platformDisabled.configureClusterInfoFromEventLog(
      coresPerExecutor = 32,
      execsPerNode = 2,
      numExecs = 8,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProviderDisabled, platformDisabled)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""
        |Spark Properties:
        |--conf spark.executor.cores=16
        |--conf spark.executor.instances=2
        |--conf spark.executor.memory=[FILL_IN_VALUE]
        |--conf spark.executor.resource.gpu.amount=1
        |--conf spark.locality.wait=0
        |--conf spark.plugins=com.nvidia.spark.SQLPlugin
        |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
        |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
        |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
        |--conf spark.rapids.sql.batchSizeBytes=1g
        |--conf spark.rapids.sql.concurrentGpuTasks=3
        |--conf spark.rapids.sql.enabled=true
        |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
        |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
        |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
        |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
        |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
        |--conf spark.sql.files.maxPartitionBytes=512m
        |--conf spark.task.resource.gpu.amount=0.001
        |
        |Comments:
        |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
        |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
        |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
        |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
        |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
        |- 'spark.rapids.sql.batchSizeBytes' was not set.
        |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
        |- 'spark.rapids.sql.enabled' was not set.
        |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
        |- 'spark.shuffle.manager' is not recommended because the Spark version on the GPU cluster is unknown during Qualification.
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
        |- $additionalSparkPluginsComment
        |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test AutoTuner honours target cluster values for dynamic allocation properties
  // enforced properties: initialExecutors and minExecutors
  // preserve properties from source: maxExecutors
  test("test AutoTuner honours target cluster values for dynamic allocation properties") {
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "20",
      "spark.executor.memory" -> "16g",
      "spark.dynamicAllocation.enabled" -> "true",
      "spark.dynamicAllocation.initialExecutors" -> "12",
      "spark.dynamicAllocation.minExecutors" -> "12",
      "spark.dynamicAllocation.maxExecutors" -> "30"
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = Map(
        "spark.dynamicAllocation.initialExecutors" -> "10",
        "spark.dynamicAllocation.minExecutors" -> "5"
      ),
      preserveSparkProperties = List(
        "spark.dynamicAllocation.maxExecutors"
      )
    )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.EMR, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 4,
      numExecs = 20,
      numExecutorNodes = 5,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.dynamicAllocation.initialExecutors=10
          |--conf spark.dynamicAllocation.maxExecutors=30
          |--conf spark.dynamicAllocation.minExecutors=5
          |--conf spark.executor.cores=16
          |--conf spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.executor.instances=10
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=13107m
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=4915m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.driver.extraJavaOptions' was not set.
          |- ${getEnforcedPropertyComment("spark.dynamicAllocation.initialExecutors")}
          |- ${getPreservedPropertyComment("spark.dynamicAllocation.maxExecutors")}
          |- ${getEnforcedPropertyComment("spark.dynamicAllocation.minExecutors")}
          |- 'spark.executor.extraJavaOptions' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- Set 'spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing driver JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- Set 'spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing executor JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $additionalSparkPluginsComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test AutoTuner bumps executor instances when dynamic allocation minExecutors is enforced
  test("test AutoTuner bumps executor instances when dynamic allocation minExecutor is enforced") {
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "20",
      "spark.executor.memory" -> "16g",
      "spark.dynamicAllocation.enabled" -> "true",
      "spark.dynamicAllocation.initialExecutors" -> "12",
      "spark.dynamicAllocation.minExecutors" -> "12"
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = Map(
        "spark.dynamicAllocation.minExecutors" -> "14"
      )
    )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.EMR, Some(targetClusterInfo))
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 4,
      numExecs = 20,
      numExecutorNodes = 5,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.dynamicAllocation.initialExecutors=14
          |--conf spark.dynamicAllocation.minExecutors=14
          |--conf spark.executor.cores=16
          |--conf spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages
          |--conf spark.executor.instances=14
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=13107m
          |--conf spark.executor.resource.gpu.amount=1
          |--conf spark.locality.wait=0
          |--conf spark.plugins=com.nvidia.spark.SQLPlugin
          |--conf spark.rapids.memory.pinnedPool.size=4915m
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=1g
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.driver.extraJavaOptions' was not set.
          |- ${getEnforcedPropertyComment("spark.dynamicAllocation.minExecutors")}
          |- 'spark.executor.extraJavaOptions' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
          |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
          |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- $shuffleManagerCommentForQualification
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- Set 'spark.driver.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing driver JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- Set 'spark.executor.extraJavaOptions=-XX:-UseTransparentHugePages' to disable Transparent Huge Pages (THP) for EMR. This recommendation does not preserve existing executor JVM options; append any additional options manually. To view the source extraJavaOptions, please refer to spark_properties.csv in the tool output.
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $additionalSparkPluginsComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test multiple scenarios with different core ratios
  private val dynamicAllocationTestCases = Table[
    Int, Int, DynamicAllocationInfo, Option[DynamicAllocationInfo]
  ](
    (
      "cpuCores", "gpuCores",
      "cpuInfo", "gpuInfoOpt"
    ),
    // Case 1: CPU cores > GPU cores: Increase number of executors
    // Source: initial=10, min=2, max=20 -> adjustRatio=2.0 -> initial=20, min=4, max=40
    (
      16, 8,
      DynamicAllocationInfo(enabled = true, "20", "2", "10"),
      Some(DynamicAllocationInfo(enabled = true, "40", "4", "20"))
    ),
    // Case 2: CPU cores < GPU cores: Decrease number of executors
    // Source: initial=8, min=4, max=16 -> adjustRatio=0.25 -> initial=2, min=1, max=4
    (
      4, 16,
      DynamicAllocationInfo(enabled = true, "16", "4", "8"),
      Some(DynamicAllocationInfo(enabled = true, "4", "1", "2"))
    ),
    // Case 3: CPU cores = GPU cores: No change in number of executors
    (
      8, 8,
      DynamicAllocationInfo(enabled = true, "12", "3", "6"),
      None
    )
  )

  forAll(dynamicAllocationTestCases) {
    (cpuCores: Int, gpuCores: Int, cpuInfo: DynamicAllocationInfo,
     gpuInfoOpt: Option[DynamicAllocationInfo]) => {
      def getMemoryGbFromCores(cores: Int, heapPerCoreGb: Double = 3.0): Int = {
        math.floor(cores * heapPerCoreGb).toInt
      }
      val execsPerNode = 2
      val testName = s"CPU $cpuCores cores to GPU $gpuCores cores"
      test(s"test dynamic allocation formula calculation for $testName") {
        val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
          "spark.executor.cores" -> cpuCores.toString,
          "spark.executor.instances" -> cpuInfo.initial,
          "spark.executor.memory" -> s"${getMemoryGbFromCores(cpuCores)}g",
          "spark.dynamicAllocation.enabled" -> "true",
          "spark.dynamicAllocation.initialExecutors" -> cpuInfo.initial,
          "spark.dynamicAllocation.minExecutors" -> cpuInfo.min,
          "spark.dynamicAllocation.maxExecutors" -> cpuInfo.max
        )

        val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
          cpuCores = Some(gpuCores),
          memoryGB = Some(getMemoryGbFromCores(gpuCores)),
          gpuCount = Some(1),
          gpuDevice = Some("a100")
        )
        val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
          logEventsProps, Some(testSparkVersion))
        val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
        platform.configureClusterInfoFromEventLog(
          coresPerExecutor = cpuCores,
          execsPerNode = execsPerNode,
          numExecs = cpuInfo.initial.toInt,
          numExecutorNodes = math.floor(cpuInfo.initial.toDouble / execsPerNode).toInt,
          sparkProperties = logEventsProps.toMap,
          systemProperties = Map.empty
        )

        val autoTuner = buildAutoTunerForTests(infoProvider, platform)
        val (properties, comments) = autoTuner.getRecommendedProperties()
        gpuInfoOpt match {
          case Some(gpuInfo) =>
            val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
            val expectedProperties = Seq(
              ("spark.dynamicAllocation.initialExecutors", gpuInfo.initial),
              ("spark.dynamicAllocation.minExecutors", gpuInfo.min),
              ("spark.dynamicAllocation.maxExecutors", gpuInfo.max)
            )
            val expectedResults = expectedProperties.map { case (k, v) => s"--conf $k=$v" } :+
              commentForDynamicAllocationAdjustment(expectedProperties.map(_._1).toList, cpuCores,
                gpuCores)
            assertExpectedLinesExist(expectedResults, autoTunerOutput)
          case None =>
            assert(!properties.exists(_.name.contains("spark.dynamicAllocation")),
              "Dynamic allocation properties should not be recommended for this test case")
        }
      }
    }
  }

  test("test dynamic allocation recommendations handle zero and negative values") {
    // Test case with edge values
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "4",
      "spark.executor.instances" -> "4",
      "spark.executor.memory" -> "16g",
      "spark.dynamicAllocation.enabled" -> "true",
      "spark.dynamicAllocation.initialExecutors" -> "0",
      "spark.dynamicAllocation.minExecutors" -> "-1",
      "spark.dynamicAllocation.maxExecutors" -> "8"
    )

    val adjustedProperties = List(
      // Only 'spark.dynamicAllocation.maxExecutors' will be adjusted
      // based on the ratio calculation since the other values are zero or negative.
      "spark.dynamicAllocation.maxExecutors"
    )

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 4,
      execsPerNode = 2,
      numExecs = 8,
      numExecutorNodes = 4,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""
         |Spark Properties:
         |--conf spark.dynamicAllocation.initialExecutors=2
         |--conf spark.dynamicAllocation.maxExecutors=2
         |--conf spark.dynamicAllocation.minExecutors=1
         |--conf spark.executor.cores=16
         |--conf spark.executor.instances=2
         |--conf spark.executor.memory=[FILL_IN_VALUE]
         |--conf spark.executor.resource.gpu.amount=1
         |--conf spark.locality.wait=0
         |--conf spark.plugins=com.nvidia.spark.SQLPlugin
         |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
         |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
         |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
         |--conf spark.rapids.sql.batchSizeBytes=1g
         |--conf spark.rapids.sql.concurrentGpuTasks=3
         |--conf spark.rapids.sql.enabled=true
         |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
         |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
         |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
         |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
         |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
         |--conf spark.sql.files.maxPartitionBytes=512m
         |--conf spark.task.resource.gpu.amount=0.001
         |
         |Comments:
         |- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
         |- 'spark.plugins' should be set to the class name required for the RAPIDS Accelerator for Apache Spark.
         |  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
         |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
         |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
         |- 'spark.rapids.sql.batchSizeBytes' was not set.
         |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
         |- 'spark.rapids.sql.enabled' was not set.
         |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
         |- 'spark.shuffle.manager' is not recommended because the Spark version on the GPU cluster is unknown during Qualification.
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
         |- $additionalSparkPluginsComment
         |- ${commentForDynamicAllocationAdjustment(adjustedProperties, 4, 16)}
         |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Source: 8 cores, 18 executors (from event log),
  // target: 16 cores (ONPREM default).
  // ConstantTotalCoresStrategy preserves total core count:
  //   ceil(8*18/16) = 9 executor instances.
  // initialExecutors is boosted to match executor.instances
  //   max(floor(8*0.5),9)=9.
  // maxExecutors is independently scaled by core ratio
  //   floor(9*0.5)=4.
  // Violation: initial(9) > max(4). Enforcement caps to 4.
  test("dynamic allocation enforces invariant " +
      "with ConstantTotalCoresStrategy") {
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "18",
      "spark.executor.memory" -> "16g",
      "spark.dynamicAllocation.enabled" -> "true",
      "spark.dynamicAllocation.initialExecutors" -> "8",
      "spark.dynamicAllocation.minExecutors" -> "4",
      "spark.dynamicAllocation.maxExecutors" -> "9"
    )

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 2,
      numExecs = 18,
      numExecutorNodes = 9,
      sparkProperties = logEventsProps.toMap,
      systemProperties = Map.empty
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) =
      autoTuner.getRecommendedProperties()
    // After enforcement: initial capped from 9 to 4,
    // max=floor(9*0.5)=4, min=max(1,floor(4*0.5))=2
    assertDynamicAllocationRecommendations(properties, comments,
      DynamicAllocationInfo(
        enabled = true, max = "4", min = "2",
        initial = "4"))
  }

  test("test CSP platform with OnPrem-style target cluster specs") {
    // Verify that CSP platforms can accept OnPrem-style target cluster specifications
    // (cpuCores/memoryGB/GPU) as a fallback when instanceType is not provided.
    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        cpuCores = Some(16),
        memoryGB = Some(64L),
        gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4))

      val appArgs = new QualificationArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        qualLogDir))

      // Qualification should run with success status
      val result = QualificationMain.mainInternal(appArgs)
      val appStatus = result.appStatus.head
      // Using hardcoded string since we do not have an enum for app statuses
      assert(appStatus.status == "SUCCESS")
    }
  }

  test("test CSP platform with invalid instance type results in failure") {
    // Verify that providing an invalid instance type on a CSP platform results in a failure.
    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        workerNodeInstanceType = Some("invalid-instance-type-99"),
        gpuCount = Some(1))

      val appArgs = new QualificationArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        qualLogDir))

      // Qualification should fail with failure status
      val result = QualificationMain.mainInternal(appArgs)
      val appStatus = result.appStatus.head
      // Using hardcoded string since we do not have an enum for app statuses
      assert(appStatus.status == "FAILURE")
      assert(appStatus.message.contains(
        classOf[MatchingInstanceTypeNotFoundException].getSimpleName))
    }
  }

  // Regression test for https://github.com/NVIDIA/spark-rapids-tools/issues/2040
  // GPU device type check should be case insensitive. Mixed-case names like "A10G"
  // should resolve correctly through the full AutoTuner pipeline.
  forAll(Table("gpuDevice", "A10G", "a10g", "T4", "t4")) { (gpuName: String) =>
    test(s"GPU device lookup is case insensitive for $gpuName") {
      val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "32g"
      )
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
        logEventsProps, Some(testSparkVersion))
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(16),
        memoryGB = Some(64L),
        gpuCount = Some(1),
        gpuDevice = Some(gpuName)
      )
      val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
        Some(targetClusterInfo))
      platform.configureClusterInfoFromEventLog(
        coresPerExecutor = 16,
        execsPerNode = 1,
        numExecs = 2,
        numExecutorNodes = 2,
        sparkProperties = logEventsProps.toMap,
        systemProperties = Map.empty
      )
      val autoTuner = buildAutoTunerForTests(infoProvider, platform)
      val (properties, _) = autoTuner.getRecommendedProperties()
      assert(properties.nonEmpty,
        s"AutoTuner should produce recommendations for GPU device '$gpuName'")
    }
  }
}
