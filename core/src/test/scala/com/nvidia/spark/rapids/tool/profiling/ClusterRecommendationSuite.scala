/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.profiling

import java.io.File
import java.nio.file.Paths

import com.nvidia.spark.rapids.tool.{GpuTypes, PlatformInstanceTypes, PlatformNames, StatusReportCounts, ToolTestUtils}
import com.nvidia.spark.rapids.tool.tuning.ProfilingAutoTunerSuiteBase
import com.nvidia.spark.rapids.tool.views.CLUSTER_INFORMATION_LABEL
import org.scalatest.prop.{TableDrivenPropertyChecks, TableFor3, TableFor4}

import org.apache.spark.sql.{SparkSession, TrampolineUtil}
import org.apache.spark.sql.rapids.tool.RecommendedClusterInfo
import org.apache.spark.sql.rapids.tool.util.FSUtils


class ClusterRecommendationSuite extends ProfilingAutoTunerSuiteBase
  with TableDrivenPropertyChecks {
  lazy val sparkSession: SparkSession = {
    SparkSession
      .builder()
      .master("local[*]")
      .appName("Rapids Spark Profiling Tool Unit Tests")
      .getOrCreate()
  }

  val validClusterRecommendationScenarios: TableFor4[
      String, String, Option[Int], RecommendedClusterInfo] = Table(
    // Define the column headers
    ("platform", "workerNodeInstanceType", "gpuCount", "expectedClusterInfo"),
    // Test scenarios
    (
      PlatformNames.DATAPROC,
      "n1-standard-16",
      Some(1),
      RecommendedClusterInfo(
        vendor = PlatformNames.DATAPROC,
        coresPerExecutor = 16,
        numWorkerNodes = 8,
        numGpusPerNode = 1,
        numExecutors = 8,
        gpuDevice = "nvidia-tesla-t4",
        dynamicAllocationEnabled = false,
        dynamicAllocationMaxExecutors = "N/A",
        dynamicAllocationMinExecutors = "N/A",
        dynamicAllocationInitialExecutors = "N/A",
        driverNodeType = Some(PlatformInstanceTypes.DEFAULT_DRIVER_NODES(PlatformNames.DATAPROC)),
        workerNodeType = Some("n1-standard-16"))
    ),
    (
      PlatformNames.DATAPROC,
      "n1-standard-32",
      Some(2),
      RecommendedClusterInfo(
        vendor = PlatformNames.DATAPROC,
        coresPerExecutor = 16,
        numWorkerNodes = 4,
        numGpusPerNode = 2,
        numExecutors = 8,
        gpuDevice = "nvidia-tesla-t4",
        dynamicAllocationEnabled = false,
        dynamicAllocationMaxExecutors = "N/A",
        dynamicAllocationMinExecutors = "N/A",
        dynamicAllocationInitialExecutors = "N/A",
        driverNodeType = Some(PlatformInstanceTypes.DEFAULT_DRIVER_NODES(PlatformNames.DATAPROC)),
        workerNodeType = Some("n1-standard-32"))
    ),
    (
      PlatformNames.DATAPROC,
      "g2-standard-16",
      None,
      RecommendedClusterInfo(
        vendor = PlatformNames.DATAPROC,
        coresPerExecutor = 16,
        numWorkerNodes = 8,
        numGpusPerNode = 1,
        numExecutors = 8,
        gpuDevice = "nvidia-l4",
        dynamicAllocationEnabled = false,
        dynamicAllocationMaxExecutors = "N/A",
        dynamicAllocationMinExecutors = "N/A",
        dynamicAllocationInitialExecutors = "N/A",
        driverNodeType = Some(PlatformInstanceTypes.DEFAULT_DRIVER_NODES(PlatformNames.DATAPROC)),
        workerNodeType = Some("g2-standard-16"))
    ),
    (
      PlatformNames.EMR,
      "g6.8xlarge",
      None,
      RecommendedClusterInfo(
        vendor = PlatformNames.EMR,
        coresPerExecutor = 32,
        numWorkerNodes = 8,
        numGpusPerNode = 1,
        numExecutors = 8,
        gpuDevice = GpuTypes.L4,
        dynamicAllocationEnabled = false,
        dynamicAllocationMaxExecutors = "N/A",
        dynamicAllocationMinExecutors = "N/A",
        dynamicAllocationInitialExecutors = "N/A",
        driverNodeType = Some(PlatformInstanceTypes.DEFAULT_DRIVER_NODES(PlatformNames.EMR)),
        workerNodeType = Some("g6.8xlarge"))
    ),
    (
      PlatformNames.DATABRICKS_AWS,
      "g5.8xlarge",
      None,
      RecommendedClusterInfo(
        vendor = PlatformNames.DATABRICKS_AWS,
        coresPerExecutor = 32,
        numWorkerNodes = 8,
        numGpusPerNode = 1,
        numExecutors = 8,
        gpuDevice = GpuTypes.A10G,
        dynamicAllocationEnabled = false,
        dynamicAllocationMaxExecutors = "N/A",
        dynamicAllocationMinExecutors = "N/A",
        dynamicAllocationInitialExecutors = "N/A",
        driverNodeType = Some(
          PlatformInstanceTypes.DEFAULT_DRIVER_NODES(PlatformNames.DATABRICKS_AWS)),
        workerNodeType = Some("g5.8xlarge"))
    )
  )
  // scalastyle:on line.size.limit
  forAll(validClusterRecommendationScenarios) {
    (platform, workerNodeInstanceType, gpuCount, expectedClusterInfo) =>
      val scenarioName = gpuCount match {
        case Some(count) => s"$workerNodeInstanceType with $count GPUs"
        case None => workerNodeInstanceType
      }
      test(s"test valid cluster shape recommendation on $platform - $scenarioName") {
        TrampolineUtil.withTempDir { tempDir =>
          val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
            tempDir.getAbsolutePath,
            workerNodeInstanceType = Some(workerNodeInstanceType),
            gpuCount = gpuCount)

          val appArgs = new ProfileArgs(Array(
            "--platform",
            platform,
            "--target-cluster-info",
            targetClusterInfoFile.toString,
            "--output-directory",
            tempDir.getAbsolutePath,
            "--csv",
            "--auto-tuner",
            s"$profilingLogDir/nds_q66_gpu.zstd"))

          val (exit, _) = ProfileMain.mainInternal(appArgs)
          assert(exit == 0)
          val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/application_1701368813061_0008")
          val fileName = CLUSTER_INFORMATION_LABEL.replace(" ", "_").toLowerCase
          val actualClusterInfoFile = Paths.get(
            s"${tempSubDir.getAbsolutePath}", s"$fileName.json"
          ).toFile
          assertRecommendedClusterInfo(actualClusterInfoFile, expectedClusterInfo)
        }
      }
  }

  test(s"test valid cluster shape recommendation on onprem") {
    val expectedClusterInfo = RecommendedClusterInfo(
      vendor = PlatformNames.ONPREM,
      coresPerExecutor = 16,
      numWorkerNodes = -1,
      numGpusPerNode = 1,
      numExecutors = 8,
      gpuDevice = GpuTypes.L4,
      dynamicAllocationEnabled = false,
      dynamicAllocationMaxExecutors = "N/A",
      dynamicAllocationMinExecutors = "N/A",
      dynamicAllocationInitialExecutors = "N/A",
      workerNodeType = Some("N/A"))
    TrampolineUtil.withTempDir { tempDir =>
      val appArgs = new ProfileArgs(Array(
        "--platform",
        PlatformNames.ONPREM,
        "--output-directory",
        tempDir.getAbsolutePath,
        "--csv",
        "--auto-tuner",
        s"$profilingLogDir/nds_q66_gpu.zstd"))

      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)
      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/application_1701368813061_0008")
      val fileName = CLUSTER_INFORMATION_LABEL.replace(" ", "_").toLowerCase
      val actualClusterInfoFile = Paths.get(
        s"${tempSubDir.getAbsolutePath}", s"$fileName.json"
      ).toFile
      assertRecommendedClusterInfo(actualClusterInfoFile, expectedClusterInfo)
    }
  }

  val invalidClusterInfoScenarios: TableFor3[String, String, Option[Int]] = Table(
    // Define the column headers
    ("platform", "instanceType", "gpuCount"),
    // Test scenarios
    (PlatformNames.DATAPROC, "n1-standard-4", None),
    (PlatformNames.EMR, "abc.8xlarge", None),
    (PlatformNames.DATABRICKS_AWS, "g5.2xlarge", Some(4))
  )

  forAll(invalidClusterInfoScenarios) {
    (platform, instanceType, gpuCount) =>
      val scenarioName = gpuCount match {
        case Some(count) => s"$instanceType with $count GPUs"
        case None => instanceType
      }
      test(s"test invalid cluster shape recommendation on $platform - $scenarioName") {
        TrampolineUtil.withTempDir { tempDir =>
          val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
            tempDir.getAbsolutePath,
            workerNodeInstanceType = Some(instanceType),
            gpuCount = gpuCount)

          val appArgs = new ProfileArgs(Array(
            "--platform",
            platform,
            "--target-cluster-info",
            targetClusterInfoFile.toString,
            "--output-directory",
            tempDir.getAbsolutePath,
            "--csv",
            "--auto-tuner",
            s"$profilingLogDir/nds_q66_gpu.zstd"))

          val (exit, _) = ProfileMain.mainInternal(appArgs)
          assert(exit == 0)
          // Status counts: 0 SUCCESS, 0 FAILURE, 0 SKIPPED, 1 UNKNOWN
          val expectedStatusCount = StatusReportCounts(0, 0, 0, 1)
          ToolTestUtils.compareStatusReport(sparkSession, expectedStatusCount,
            s"${tempDir.getAbsolutePath}/${Profiler.SUBDIR}/profiling_status.csv")
        }
      }
  }


  /**
   * Test to validate the cluster shape recommendation with enforced spark properties.
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
   *    spark.executor.memory: 12g
   *    spark.memory.offHeap.enabled: true
   *    spark.memory.offHeap.size: 2g
   *    spark.sql.shuffle.partitions: 400
   * }}}
   */
  test(s"test valid cluster shape recommendation with enforced spark properties on dataproc") {
    val expectedClusterInfo = RecommendedClusterInfo(
      vendor = PlatformNames.DATAPROC,
      coresPerExecutor = 8,
      numWorkerNodes = 2,
      numGpusPerNode = 1,
      numExecutors = 2,
      gpuDevice = "nvidia-l4",
      dynamicAllocationEnabled = false,
      dynamicAllocationMaxExecutors = "N/A",
      dynamicAllocationMinExecutors = "N/A",
      dynamicAllocationInitialExecutors = "N/A",
      driverNodeType = Some("n1-standard-8"),
      workerNodeType = Some("g2-standard-8")
    )
    val expectedEnforcedSparkProperties = Map(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "12g",
      "spark.memory.offHeap.enabled" -> "true",
      "spark.memory.offHeap.size" -> "2g",
      "spark.sql.shuffle.partitions" -> "400"
    )
    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        driverNodeInstanceType = expectedClusterInfo.driverNodeType,
        workerNodeInstanceType = expectedClusterInfo.workerNodeType,
        enforcedSparkProperties = expectedEnforcedSparkProperties)

      val appArgs = new ProfileArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        "--csv",
        "--auto-tuner",
        s"$profilingLogDir/gpu_oom_eventlog.zstd"))

      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)
      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/app-20250305192829-0000")
      val fileName = CLUSTER_INFORMATION_LABEL.replace(" ", "_").toLowerCase
      val actualClusterInfoFile = Paths.get(
        s"${tempSubDir.getAbsolutePath}", s"$fileName.json"
      ).toFile

      // 1. Verify the recommended cluster info
      assertRecommendedClusterInfo(actualClusterInfoFile, expectedClusterInfo)

      // 2. Verify the enforced spark properties
      val logFile = getOutputFilePath(tempDir, "profile.log")
      val profileLogContent = FSUtils.readFileContentAsUTF8(logFile)
      val actualResults = extractAutoTunerResults(profileLogContent)

      // scalastyle:off line.size.limit
      val expectedResults =
        s"""|
            |Spark Properties:
            |--conf spark.dataproc.enhanced.execution.enabled=false
            |--conf spark.dataproc.enhanced.optimizer.enabled=false
            |--conf spark.executor.cores=8
            |--conf spark.executor.instances=2
            |--conf spark.executor.memory=12g
            |--conf spark.memory.offHeap.enabled=true
            |--conf spark.memory.offHeap.size=2g
            |--conf spark.rapids.memory.pinnedPool.size=5325m
            |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
            |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
            |--conf spark.rapids.sql.batchSizeBytes=2147483647b
            |--conf spark.rapids.sql.concurrentGpuTasks=3
            |--conf spark.rapids.sql.enabled=true
            |--conf spark.rapids.sql.multiThreadedRead.numThreads=40
            |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
            |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
            |--conf spark.sql.shuffle.partitions=400
            |--conf spark.task.resource.gpu.amount=0.001
            |
            |Comments:
            |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
            |- 'spark.dataproc.enhanced.execution.enabled' was not set.
            |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
            |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
            |- ${getEnforcedPropertyComment("spark.executor.cores")}
            |- 'spark.executor.instances' was not set.
            |- ${getEnforcedPropertyComment("spark.executor.memory")}
            |- ${getEnforcedPropertyComment("spark.memory.offHeap.enabled")}
            |- ${getEnforcedPropertyComment("spark.memory.offHeap.size")}
            |- 'spark.rapids.memory.pinnedPool.size' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
            |- 'spark.rapids.sql.batchSizeBytes' was not set.
            |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
            |- 'spark.rapids.sql.enabled' was not set.
            |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
            |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
            |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
            |- ${getEnforcedPropertyComment("spark.sql.shuffle.partitions")}
            |- $shufflePartitionsCommentForSpilling
            |- ${classPathComments("rapids.jars.outdated")}
            |- ${classPathComments("rapids.shuffle.jars")}
            |""".stripMargin.trim
      // scalastyle:on line.size.limit
      compareOutput(expectedResults, actualResults)
    }
  }

  /**
   * Test to validate the cluster shape recommendation with enforced spark properties.
   * This tests that if the user has enforced `spark.executor.instances`, this will
   * affect the recommended cluster shape.
   *
   * Target Cluster YAML file:
   * {{{
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
      driverNodeType = Some(PlatformInstanceTypes.DEFAULT_DRIVER_NODES(PlatformNames.DATAPROC)),
      workerNodeType = Some("g2-standard-8")
    )
    val expectedEnforcedSparkProperties = Map(
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "4",
      "spark.executor.memory" -> "12g"
    )
    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        driverNodeInstanceType = expectedClusterInfo.driverNodeType,
        workerNodeInstanceType = expectedClusterInfo.workerNodeType,
        enforcedSparkProperties = expectedEnforcedSparkProperties)

      val appArgs = new ProfileArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        "--csv",
        "--auto-tuner",
        s"$profilingLogDir/gpu_oom_eventlog.zstd"))

      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)
      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/app-20250305192829-0000")
      val fileName = CLUSTER_INFORMATION_LABEL.replace(" ", "_").toLowerCase
      val actualClusterInfoFile = Paths.get(
        s"${tempSubDir.getAbsolutePath}", s"$fileName.json"
      ).toFile

      // 1. Verify the recommended cluster info
      assertRecommendedClusterInfo(actualClusterInfoFile, expectedClusterInfo)

      // 2. Verify the enforced spark properties
      val logFile = getOutputFilePath(tempDir, "profile.log")
      val profileLogContent = FSUtils.readFileContentAsUTF8(logFile)
      val actualResults = extractAutoTunerResults(profileLogContent)

      // scalastyle:off line.size.limit
      val expectedResults =
        s"""|
            |Spark Properties:
            |--conf spark.dataproc.enhanced.execution.enabled=false
            |--conf spark.dataproc.enhanced.optimizer.enabled=false
            |--conf spark.executor.cores=8
            |--conf spark.executor.instances=4
            |--conf spark.executor.memory=12g
            |--conf spark.rapids.memory.pinnedPool.size=6349m
            |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
            |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
            |--conf spark.rapids.sql.batchSizeBytes=2147483647b
            |--conf spark.rapids.sql.concurrentGpuTasks=3
            |--conf spark.rapids.sql.enabled=true
            |--conf spark.rapids.sql.multiThreadedRead.numThreads=40
            |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
            |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
            |--conf spark.sql.shuffle.partitions=400
            |--conf spark.task.resource.gpu.amount=0.001
            |
            |Comments:
            |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
            |- 'spark.dataproc.enhanced.execution.enabled' was not set.
            |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
            |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
            |- ${getEnforcedPropertyComment("spark.executor.cores")}
            |- ${getEnforcedPropertyComment("spark.executor.instances")}
            |- ${getEnforcedPropertyComment("spark.executor.memory")}
            |- 'spark.rapids.memory.pinnedPool.size' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
            |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
            |- 'spark.rapids.sql.batchSizeBytes' was not set.
            |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
            |- 'spark.rapids.sql.enabled' was not set.
            |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
            |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
            |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
            |- $shufflePartitionsCommentForSpilling
            |- ${classPathComments("rapids.jars.outdated")}
            |- ${classPathComments("rapids.shuffle.jars")}
            |""".stripMargin.trim
      // scalastyle:on line.size.limit
      compareOutput(expectedResults, actualResults)
    }
  }

  test("test CSP platform with OnPrem-style target cluster specs") {
    // Verify that CSP platforms can accept OnPrem-style target cluster specifications
    // (cpuCores/memoryGB/GPU) as a fallback when instanceType is not provided.
    //
    // Expected behavior:
    // - workerNodeType: "N/A" because the target cluster uses OnPrem-style specs (cpuCores,
    //   memoryGB, GPU) instead of a CSP instance type
    // - driverNodeType: platform default for dataproc
    val expectedClusterInfo = RecommendedClusterInfo(
      vendor = PlatformNames.DATAPROC,
      coresPerExecutor = 16,
      numWorkerNodes = 8,
      numGpusPerNode = 1,
      numExecutors = 8,
      gpuDevice = "nvidia-l4",
      dynamicAllocationEnabled = false,
      dynamicAllocationMaxExecutors = "N/A",
      dynamicAllocationMinExecutors = "N/A",
      dynamicAllocationInitialExecutors = "N/A",
      driverNodeType = Some(PlatformInstanceTypes.DEFAULT_DRIVER_NODES(PlatformNames.DATAPROC)),
      workerNodeType = Some("N/A"))

    TrampolineUtil.withTempDir { tempDir =>
      val targetClusterInfoFile = ToolTestUtils.createTargetClusterInfoFile(
        tempDir.getAbsolutePath,
        cpuCores = Some(16),
        memoryGB = Some(64L),
        gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4))

      val appArgs = new ProfileArgs(Array(
        "--platform",
        PlatformNames.DATAPROC,
        "--target-cluster-info",
        targetClusterInfoFile.toString,
        "--output-directory",
        tempDir.getAbsolutePath,
        "--csv",
        "--auto-tuner",
        s"$profilingLogDir/nds_q66_gpu.zstd"))

      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)

      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/application_1701368813061_0008")
      val fileName = CLUSTER_INFORMATION_LABEL.replace(" ", "_").toLowerCase
      val actualClusterInfoFile = Paths.get(
        s"${tempSubDir.getAbsolutePath}", s"$fileName.json"
      ).toFile
      assertRecommendedClusterInfo(actualClusterInfoFile, expectedClusterInfo)
    }
  }
}
