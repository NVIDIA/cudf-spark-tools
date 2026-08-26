/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

import scala.annotation.tailrec
import scala.collection.mutable

import com.nvidia.spark.rapids.tool.planparser.db.DatabricksParseHelper
import com.nvidia.spark.rapids.tool.tuning.{SparkMaster, TargetClusterProps, TuningEntryTrait, WorkerInfo}

import org.apache.spark.internal.Logging
import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.{MatchingInstanceTypeNotFoundException, RecommendedClusterInfo, SourceClusterInfo}
import org.apache.spark.sql.rapids.tool.util.{SparkRuntime, StringUtils}

/**
 *  Utility object containing constants for various platform names.
 */
object PlatformNames {
  val DATABRICKS_AWS = "databricks-aws"
  val DATABRICKS_AZURE = "databricks-azure"
  val DATAPROC = "dataproc"
  val DATAPROC_GKE = "dataproc-gke"
  val DATAPROC_SL = "dataproc-serverless"
  val EMR = "emr"
  val ONPREM = "onprem"
  val DEFAULT: String = ONPREM

  /**
   * Return a list of all supported platform names.
   */
  def getAllNames: List[String] = List(
    DATABRICKS_AWS, s"$DATABRICKS_AWS-$A10GGpu", s"$DATABRICKS_AWS-$T4Gpu",
    DATABRICKS_AZURE, s"$DATABRICKS_AZURE-$T4Gpu",
    DATAPROC, s"$DATAPROC-$L4Gpu", s"$DATAPROC-$T4Gpu",
    DATAPROC_GKE, s"$DATAPROC_GKE-$L4Gpu", s"$DATAPROC_GKE-$T4Gpu",
    DATAPROC_SL, s"$DATAPROC_SL-$L4Gpu",
    EMR, s"$EMR-$A10Gpu", s"$EMR-$A10GGpu", s"$EMR-$T4Gpu",
    ONPREM, s"$ONPREM-$A100Gpu"
  )
}

case class DynamicAllocationInfo(enabled: Boolean, max: String, min: String, initial: String)

/**
 * Case class representing the key for the node instance map.
 * @param instanceType Name of the instance type
 * @param gpuCount Count of GPUs on the instance. This is required for certain
 *                 cases where a given instance type can have multiple GPU counts
 *                 (e.g. n1-standard-16 with 1, 2, or 4 GPUs).
 */
case class NodeInstanceMapKey(instanceType: String, gpuCount: Option[Int] = None)
  extends Ordered[NodeInstanceMapKey] {
  override def toString: String = gpuCount match {
    case Some(count) => s"NodeInstanceMapKey(instanceType='$instanceType', gpuCount=$count)"
    case None => s"NodeInstanceMapKey(instanceType='$instanceType')"
  }

  /**
   * Defines the ordering by `instanceType` lexicographically then by `gpuCount` numerically
   */
  override def compare(that: NodeInstanceMapKey): Int = {
    val instanceCmp = this.instanceType.compareTo(that.instanceType)
    if (instanceCmp != 0) {
      instanceCmp
    } else {
      gpuCount.getOrElse(0) - that.gpuCount.getOrElse(0)
    }
  }
}

// resource information and name of the CSP instance types, or for onprem its
// the executor information since we can't recommend node types
case class InstanceInfo(cores: Int, memoryMB: Long, name: String,
      numGpus: Int, gpuDevice: GpuDevice) {
  /**
   * Get the memory per executor based on the instance type.
   * Note: For GPU instances, num of executors is same as the number of GPUs.
   */
  def getMemoryPerExec: Double = {
    memoryMB.toDouble / numGpus
  }
}

object InstanceInfo {
  def createDefaultInstance(cores: Int, memoryMB: Long,
      numGpus: Int, gpuDevice: GpuDevice): InstanceInfo = {
    InstanceInfo(cores, memoryMB, "N/A", numGpus, gpuDevice)
  }
}

// This is a map of the instance types to the InstanceInfo.
// format (instance type name, gpu count) -> InstanceInfo about that CSP node instance type
object PlatformInstanceTypes {

  // Default driver node mappings for each platform
  // Selection: Nodes having atleast 8 cores and 32GB memory
  val DEFAULT_DRIVER_NODES: Map[String, String] = Map(
    PlatformNames.DATABRICKS_AWS -> "m6gd.2xlarge",
    PlatformNames.DATABRICKS_AZURE -> "Standard_E8ds_v4",
    PlatformNames.DATAPROC -> "n1-standard-16",
    PlatformNames.DATAPROC_GKE -> "n1-standard-16",
    PlatformNames.DATAPROC_SL -> "n1-standard-16",
    PlatformNames.EMR -> "i3.2xlarge"
  )

  // Default worker node mappings for each platform
  // Selection: Nodes used by the estimation model
  val DEFAULT_WORKER_NODES: Map[String, String] = Map(
    PlatformNames.DATABRICKS_AWS -> "g5.8xlarge",
    PlatformNames.DATABRICKS_AZURE -> "Standard_NC8as_T4_v3",
    PlatformNames.DATAPROC -> "g2-standard-16",
    PlatformNames.DATAPROC_GKE -> "g2-standard-16",
    PlatformNames.DATAPROC_SL -> "g2-standard-16",
    PlatformNames.EMR -> "g6.4xlarge"
  )

  // Using G6 instances for EMR
  val EMR_BY_INSTANCE_NAME: Map[NodeInstanceMapKey, InstanceInfo] = Map(
    NodeInstanceMapKey("g6.xlarge") -> InstanceInfo(4, 16 * 1024, "g6.xlarge", 1, L4Gpu),
    NodeInstanceMapKey("g6.2xlarge") -> InstanceInfo(8, 32 * 1024, "g6.2xlarge", 1, L4Gpu),
    NodeInstanceMapKey("g6.4xlarge") -> InstanceInfo(16, 64 * 1024, "g6.4xlarge", 1, L4Gpu),
    NodeInstanceMapKey("g6.8xlarge") -> InstanceInfo(32, 128 * 1024, "g6.8xlarge", 1, L4Gpu),
    NodeInstanceMapKey("g6.12xlarge") -> InstanceInfo(48, 192 * 1024, "g6.12xlarge", 4, L4Gpu),
    NodeInstanceMapKey("g6.16xlarge") -> InstanceInfo(64, 256 * 1024, "g6.16xlarge", 1, L4Gpu)
  )

  // Using G5 instances. To be updated once G6 availability on Databricks
  // is consistent
  val DATABRICKS_AWS_BY_INSTANCE_NAME: Map[NodeInstanceMapKey, InstanceInfo] = Map(
    NodeInstanceMapKey("g5.xlarge") -> InstanceInfo(4, 16 * 1024, "g5.xlarge", 1, A10GGpu),
    NodeInstanceMapKey("g5.2xlarge") -> InstanceInfo(8, 32 * 1024, "g5.2xlarge", 1, A10GGpu),
    NodeInstanceMapKey("g5.4xlarge") -> InstanceInfo(16, 64 * 1024, "g5.4xlarge", 1, A10GGpu),
    NodeInstanceMapKey("g5.8xlarge") -> InstanceInfo(32, 128 * 1024, "g5.8xlarge", 1, A10GGpu),
    NodeInstanceMapKey("g5.12xlarge") -> InstanceInfo(48, 192 * 1024, "g5.12xlarge", 4, A10GGpu),
    NodeInstanceMapKey("g5.16xlarge") -> InstanceInfo(64, 256 * 1024, "g5.16xlarge", 1, A10GGpu)
  )

  // Standard_NC4as_T4_v3 - only recommending nodes with T4's for now, add more later
  val AZURE_NCAS_T4_V3_BY_INSTANCE_NAME: Map[NodeInstanceMapKey, InstanceInfo] = Map(
    NodeInstanceMapKey("Standard_NC4as_T4_v3") ->
      InstanceInfo(4, 28 * 1024, "Standard_NC4as_T4_v3", 1, T4Gpu), // 1 GPU
    NodeInstanceMapKey("Standard_NC8as_T4_v3") ->
      InstanceInfo(8, 56 * 1024, "Standard_NC8as_T4_v3", 1, T4Gpu), // 1 GPU
    NodeInstanceMapKey("Standard_NC16as_T4_v3") ->
      InstanceInfo(16, 110 * 1024, "Standard_NC16as_T4_v3", 1, T4Gpu), // 1 GPU
    NodeInstanceMapKey("Standard_NC64as_T4_v3") ->
      InstanceInfo(64, 440 * 1024, "Standard_NC64as_T4_v3", 4, T4Gpu)  // 4 GPUs
  )

  // dataproc and dataproc-gke
  // Google supports 1, 2, or 4 Gpus of most types
  // added to n1-standard boxes. You may be able to add 8 v100's but we
  // are going to ignore that.
  val DATAPROC_BY_INSTANCE_NAME: Map[NodeInstanceMapKey, InstanceInfo] = Map(
    // g2-standard instances with Intel GPUs
    NodeInstanceMapKey("g2-standard-4") ->
      InstanceInfo(4, 16 * 1024, "g2-standard-4", 1, L4Gpu),
    NodeInstanceMapKey("g2-standard-8") ->
      InstanceInfo(8, 32 * 1024, "g2-standard-8", 1, L4Gpu),
    NodeInstanceMapKey("g2-standard-12") ->
      InstanceInfo(12, 48 * 1024, "g2-standard-12", 1, L4Gpu),
    NodeInstanceMapKey("g2-standard-16") ->
      InstanceInfo(16, 64 * 1024, "g2-standard-16", 1, L4Gpu),
    NodeInstanceMapKey("g2-standard-24") ->
      InstanceInfo(24, 96 * 1024, "g2-standard-24", 2, L4Gpu),
    NodeInstanceMapKey("g2-standard-32") ->
      InstanceInfo(32, 128 * 1024, "g2-standard-32", 1, L4Gpu),
    NodeInstanceMapKey("g2-standard-48") ->
      InstanceInfo(48, 192 * 1024, "g2-standard-48", 4, L4Gpu),
    NodeInstanceMapKey("g2-standard-96") ->
      InstanceInfo(96, 384 * 1024, "g2-standard-96", 8, L4Gpu),
    // Existing n1-standard instances
    NodeInstanceMapKey(instanceType = "n1-standard-1", gpuCount = Some(1)) ->
      InstanceInfo(1, 1 * 3840, "n1-standard-1", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-2", gpuCount = Some(1)) ->
      InstanceInfo(2, 2 * 3840, "n1-standard-2", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-4", gpuCount = Some(1)) ->
      InstanceInfo(4, 4 * 3840, "n1-standard-4", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-8", gpuCount = Some(1)) ->
      InstanceInfo(8, 8 * 3840, "n1-standard-8", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-16", gpuCount = Some(1)) ->
      InstanceInfo(16, 16 * 3840, "n1-standard-16", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-32", gpuCount = Some(1)) ->
      InstanceInfo(32, 32 * 3840, "n1-standard-32", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-64", gpuCount = Some(1)) ->
      InstanceInfo(64, 64 * 3840, "n1-standard-64", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-96", gpuCount = Some(1)) ->
      InstanceInfo(96, 96 * 3840, "n1-standard-96", 1, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-1", gpuCount = Some(2)) ->
      InstanceInfo(1, 1 * 3840, "n1-standard-1", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-2", gpuCount = Some(2)) ->
      InstanceInfo(2, 2 * 3840, "n1-standard-2", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-4", gpuCount = Some(2)) ->
      InstanceInfo(4, 4 * 3840, "n1-standard-4", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-8", gpuCount = Some(2)) ->
      InstanceInfo(8, 8 * 3840, "n1-standard-8", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-16", gpuCount = Some(2)) ->
      InstanceInfo(16, 16 * 3840, "n1-standard-16", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-32", gpuCount = Some(2)) ->
      InstanceInfo(32, 32 * 3840, "n1-standard-32", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-64", gpuCount = Some(2)) ->
      InstanceInfo(64, 64 * 3840, "n1-standard-64", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-96", gpuCount = Some(2)) ->
      InstanceInfo(96, 96 * 3840, "n1-standard-96", 2, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-1", gpuCount = Some(4)) ->
      InstanceInfo(1, 1 * 3840, "n1-standard-1", 4, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-2", gpuCount = Some(4)) ->
      InstanceInfo(2, 2 * 3840, "n1-standard-2", 4, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-4", gpuCount = Some(4)) ->
      InstanceInfo(4, 4 * 3840, "n1-standard-4", 4, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-8", gpuCount = Some(4)) ->
      InstanceInfo(8, 8 * 3840, "n1-standard-8", 4, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-16", gpuCount = Some(4)) ->
      InstanceInfo(16, 16 * 3840, "n1-standard-16", 4, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-32", gpuCount = Some(4)) ->
      InstanceInfo(32, 32 * 3840, "n1-standard-32", 4, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-64", gpuCount = Some(4)) ->
      InstanceInfo(64, 64 * 3840, "n1-standard-64", 4, T4Gpu),
    NodeInstanceMapKey(instanceType = "n1-standard-96", gpuCount = Some(4)) ->
      InstanceInfo(96, 96 * 3840, "n1-standard-96", 4, T4Gpu)
  )
}

/**
 * Represents a platform and its associated recommendations.
 *
 * @param gpuDevice Gpu Device present in the platform
 * @param targetCluster Target cluster information (e.g. instance type, GPU type)
 */
abstract class Platform(var gpuDevice: Option[GpuDevice],
    val targetCluster: Option[TargetClusterProps]) extends Logging {
  def platformName: String
  private def defaultRecommendedWorkerNode: Option[NodeInstanceMapKey] = {
    PlatformInstanceTypes.DEFAULT_WORKER_NODES.get(platformName)
      .map(instanceType => NodeInstanceMapKey(instanceType = instanceType))
  }
  private def defaultRecommendedDriverNode: Option[NodeInstanceMapKey] = {
    PlatformInstanceTypes.DEFAULT_DRIVER_NODES.get(platformName)
      .map(instanceType => NodeInstanceMapKey(instanceType = instanceType))
  }
  private def defaultRecommendedCoresPerExec: Int = 16
  def defaultGpuDevice: GpuDevice = T4Gpu
  def defaultNumGpus: Int = 1

  /**
   * Fraction of the system's total memory reserved by resource managers (e.g., YARN).
   * Available memory for executors = total * (1 - nonExecutorMemoryFraction).
   * For example, if YARN reserves 20% of memory, this value should be 0.2.
   */
  def nonExecutorMemoryFraction: Double

  val sparkVersionLabel: String = "Spark version"

  /**
  * Returns the recommended instance info based on the provided WorkerInfo.
  * See [[OnPremPlatform.getRecommendedInstanceInfoFromTargetWorker]] for OnPrem.
  *
  * Behavior for CSP platforms:
  * - If a specific instance type is provided and found in the platform's instance map,
  *   returns the corresponding InstanceInfo.
  * - If the specified instance type is not found, throws a MatchingInstanceTypeNotFoundException.
  * - If no instance type is specified, tries to use OnPrem-style specs (cpuCores/memoryGB/GPU).
  * - If neither instanceType nor OnPrem-style specs are provided, falls back to the platform's
  *   default recommended instance type.
  */
  protected def getRecommendedInstanceInfoFromTargetWorker(
      workerInfoOpt: Option[WorkerInfo]): Option[InstanceInfo] = {
    // Helper function to get default instance info
    def getDefaultInstance: Option[InstanceInfo] = {
      val defaultInstanceInfo = defaultRecommendedWorkerNode.flatMap(getInstanceMapByName.get)
      logInfo("Worker info is not provided in the target cluster. " +
        s"Using default instance type: $defaultInstanceInfo")
      defaultInstanceInfo
    }

    workerInfoOpt match {
      case Some(workerInfo) =>
        // Try CSP-style first (instanceType)
        workerInfo.getNodeInstanceMapKey match {
          case Some(nodeInstanceMapKey) =>
            getInstanceMapByName.get(nodeInstanceMapKey).orElse {
              val errorMsg =
                s"""
                   |Could not find matching instance type in resources map.
                   |Requested: $nodeInstanceMapKey
                   |Supported: ${getInstanceMapByName.keys.toSeq.sorted.mkString(", ")}
                   |
                   |Next Steps:
                   |Update the target cluster YAML with a valid instance type and GPU count, or skip
                   |it to use default: ${defaultRecommendedWorkerNode.getOrElse("None")}
                   |""".stripMargin.trim
              throw new MatchingInstanceTypeNotFoundException(errorMsg)
            }
          case None =>
            // Try OnPrem-style (cpuCores/memoryGB/GPU)
            if (workerInfo.isOnpremInfo) {
              workerInfo.getGpu.device.map { gpuDevice =>
                InstanceInfo.createDefaultInstance(
                  cores = workerInfo.cpuCores,
                  memoryMB = workerInfo.memoryGB * 1024L,
                  numGpus = workerInfo.getGpu.count,
                  gpuDevice = gpuDevice)
              }
            } else {
              // Neither instanceType nor OnPrem-style provided, use default
              getDefaultInstance
            }
        }
      case None =>
        // No workerInfo provided, use default
        getDefaultInstance
    }
  }

  // It's not ideal to use vars here but to minimize changes and
  // keep backwards compatibility we put them here for now and hopefully
  // in future we can refactor.
  var clusterInfoFromEventLog: Option[SourceClusterInfo] = None
  // instance information for the gpu node type we will use to run with
  var recommendedWorkerNode: Option[InstanceInfo] = {
    getRecommendedInstanceInfoFromTargetWorker(targetCluster.map(_.getWorkerInfo))
  }

  // overall final recommended cluster configuration
  var recommendedClusterInfo: Option[RecommendedClusterInfo] = None

  // Default recommendation based on NDS benchmarks (note: this could be platform specific)
  final def recommendedCoresPerExec: Int = {
    recommendedWorkerNode.map(instInfo => instInfo.cores / instInfo.numGpus)
      .getOrElse(defaultRecommendedCoresPerExec)
  }

  final def recommendedGpuDevice: GpuDevice = {
    recommendedWorkerNode.map(_.gpuDevice).getOrElse(defaultGpuDevice)
  }

  final def recommendedNumGpus: Int = {
    recommendedWorkerNode.map(_.numGpus).getOrElse(defaultNumGpus)
  }

  // Default runtime for the platform
  val defaultRuntime: SparkRuntime.SparkRuntime = SparkRuntime.SPARK
  // Set of supported runtimes for the platform
  protected val supportedRuntimes: Set[SparkRuntime.SparkRuntime] = Set(
    SparkRuntime.SPARK, SparkRuntime.SPARK_RAPIDS, SparkRuntime.AURON
  )

  // scalastyle:off line.size.limit
  // Supported Spark version to RapidsShuffleManager version mapping.
  // Reference: https://docs.nvidia.com/spark-rapids/user-guide/latest/additional-functionality/rapids-shuffle.html#rapids-shuffle-manager
  // TODO: Issue to automate this https://github.com/NVIDIA/cudf-spark-tools/issues/1676
  // scalastyle:on line.size.limit
  val supportedShuffleManagerVersionMap: Array[(String, String)] = Array(
    "3.2.0" -> "320",
    "3.2.1" -> "321",
    "3.2.2" -> "322",
    "3.2.3" -> "323",
    "3.2.4" -> "324",
    "3.3.0" -> "330",
    "3.3.1" -> "331",
    "3.3.2" -> "332",
    "3.3.3" -> "333",
    "3.3.4" -> "334",
    "3.4.0" -> "340",
    "3.4.1" -> "341",
    "3.4.2" -> "342",
    "3.4.3" -> "343",
    "3.5.0" -> "350",
    "3.5.1" -> "351",
    "3.5.2" -> "352",
    "3.5.3" -> "353",
    "3.5.4" -> "354",
    "3.5.5" -> "355",
    "3.5.6" -> "356",
    "3.5.7" -> "357",
    "4.0.0" -> "400"
  )

  /**
   * Determine the appropriate RapidsShuffleManager version based on the
   * provided spark version.
   */
  def getShuffleManagerVersion(sparkVersion: String): Option[String] = {
    supportedShuffleManagerVersionMap.collectFirst {
      case (supportedVersion, smVersion) if sparkVersion.contains(supportedVersion) => smVersion
    }
  }

  /**
   * Identify the latest supported Spark and RapidsShuffleManager version for the platform.
   */
  lazy val latestSupportedShuffleManagerInfo: (String, String) = {
    supportedShuffleManagerVersionMap.maxBy(_._1)
  }

  /**
   * Checks if the given runtime is supported by the platform.
   */
  def isRuntimeSupported(runtime: SparkRuntime.SparkRuntime): Boolean = {
    supportedRuntimes.contains(runtime)
  }

  // This function allow us to have one gpu type used by the auto
  // tuner recommendations but have a different GPU used for speedup
  // factors since we don't have speedup factors for all combinations of
  // platforms and GPUs. We expect speedup factor usage to be going away
  // so this is less of an issue.
  def defaultGpuForSpeedupFactor: GpuDevice = defaultGpuDevice

  /**
   * Recommendations to be excluded from the list of recommendations.
   * These have the highest priority.
   */
  val recommendationsToExclude: Set[String] = Set.empty

  /**
   * Tuning table entries that are disabled by default but should be enabled for this platform.
   * This provides platform-level scoping for tuning entries: entries can be defined in the
   * tuning table with `enabled: false` and selectively enabled by the platforms that need them.
   */
  val platformEnabledTuningEntries: Set[String] = Set.empty

  /**
   * Platform-specific recommendations that should be included in the final list of recommendations.
   * For example: we used to set "spark.databricks.optimizer.dynamicFilePruning" to false for the
   *              Databricks platform.
   */
  val platformSpecificRecommendations: Map[String, String] = Map.empty

  /**
   * User-enforced recommendations that should be included in the final list of recommendations.
   */
  lazy val userEnforcedRecommendations: Map[String, String] = {
    targetCluster.map(_.getSparkProperties.enforcedPropertiesMap).getOrElse(Map.empty)
  }

  /**
   * Dynamically calculates the recommendation for a specific Spark property by invoking
   * the appropriate function based on `sparkProperty`.
   * TODO: Implement this function and integrate with existing code in AutoTuner
   *
   * @param sparkProperty The Spark property for which the recommendation is calculated.
   * @param args Variable list of arguments passed to the calculation function for dynamic
   *             processing.
   * @return Optional string containing the recommendation, or `None` if unavailable.
   */
  def getRecommendation(sparkProperty: String, args: Any*): Option[String] = None

  /**
   * Checks if the `property` is valid:
   * 1. It should not be in exclusion list
   *   OR
   * 2. It should be in the inclusion list
   */
  def isValidRecommendation(property: String): Boolean = {
    !recommendationsToExclude.contains(property) ||
      platformSpecificRecommendations.contains(property) ||
      userEnforcedRecommendations.contains(property)
  }

  /**
   * Checks if the `comment` is valid:
   * 1. It should not have any property from the exclusion list
   */
  def isValidComment(comment: String): Boolean = {
    recommendationsToExclude.forall(excluded => !comment.contains(excluded))
  }

  def getOperatorScoreFile: String = {
    s"operatorsScore-$platformName-$getGpuOrDefaultForSpeedupFactors.csv"
  }

  def getDefaultOperatorScoreFile: String = {
    s"operatorsScore-$platformName-$defaultGpuForSpeedupFactor.csv"
  }

  final def getGpuOrDefault: GpuDevice = gpuDevice.getOrElse(defaultGpuDevice)

  final def getGpuOrDefaultForSpeedupFactors: GpuDevice =
    gpuDevice.getOrElse(defaultGpuForSpeedupFactor)

  final def setGpuDevice(gpuDevice: GpuDevice): Unit = {
    this.gpuDevice = Some(gpuDevice)
  }

  /**
   * Get the platform-specific GPU device name.
   * By default, returns the GPU device's string representation.
   * Override this method in platform-specific classes to provide
   * platform-specific GPU device naming (e.g., "nvidia-l4" for Dataproc).
   *
   * @param gpuDevice The GPU device to convert to platform-specific name
   * @return Platform-specific GPU device name
   */
  def getPlatformGpuDeviceName(gpuDevice: GpuDevice): String = {
    gpuDevice.toString
  }

  /**
   * Important system properties that should be retained based on platform.
   */
  def getRetainedSystemProps: Set[String] = Set.empty

  def getExecutorHeapMemoryMB(sparkPropertiesFn: String => Option[String]): Long = {
    val executorMemoryFromConf = sparkPropertiesFn("spark.executor.memory")
    if (executorMemoryFromConf.isDefined) {
      StringUtils.convertToMB(executorMemoryFromConf.getOrElse("0"), Some(ByteUnit.BYTE))
    } else {
      // TODO: Potentially enhance this to handle if no config then check the executor
      //  added or resource profile added events for the heap size
      val sparkMaster = SparkMaster(sparkPropertiesFn("spark.master"))
      sparkMaster.map(_.defaultExecutorMemoryMB).getOrElse(0L)
    }
  }

  def getSparkOffHeapMemoryMB(sparkPropertiesFn: String => Option[String]): Option[Long] = {
    // Return if offHeap is not enabled
    if (!sparkPropertiesFn("spark.memory.offHeap.enabled").getOrElse("false").toBoolean) {
      return None
    }

    sparkPropertiesFn("spark.memory.offHeap.size").map { size =>
      StringUtils.convertToMB(size, Some(ByteUnit.BYTE))
    }
  }

  def getPySparkMemoryMB(sparkPropertiesFn: String => Option[String]): Option[Long] = {
    // Avoiding explicitly checking if it is PySpark app, if the user has set
    // the memory for PySpark, we will use that.
    sparkPropertiesFn("spark.executor.pyspark.memory").map {
      size => StringUtils.convertToMB(size, Some(ByteUnit.MiB))
    }
  }

  def getExecutorOverheadMemoryMB(sparkPropertiesFn: String => Option[String]): Long = {
    val executorMemoryOverheadFromConf = sparkPropertiesFn("spark.executor.memoryOverhead")
    val execMemOverheadFactorFromConf = sparkPropertiesFn("spark.executor.memoryOverheadFactor")
    val execHeapMemoryMB = getExecutorHeapMemoryMB(sparkPropertiesFn)
    if (executorMemoryOverheadFromConf.isDefined) {
      StringUtils.convertToMB(executorMemoryOverheadFromConf.get, Some(ByteUnit.MiB))
    } else if (execHeapMemoryMB > 0) {
      if (execMemOverheadFactorFromConf.isDefined) {
        (execHeapMemoryMB * execMemOverheadFactorFromConf.get.toDouble).toLong
      } else {
        val sparkMasterConf = sparkPropertiesFn("spark.master")
        sparkMasterConf match {
          case None => 0L
          case Some(sparkMaster) =>
            if (sparkMaster.contains("yarn")) {
              // default is 10%
              (execHeapMemoryMB * 0.1).toLong
            } else if (sparkMaster.contains("k8s")) {
              val k8sOverheadFactor = sparkPropertiesFn("spark.kubernetes.memoryOverheadFactor")
              if (k8sOverheadFactor.isDefined) {
                (execHeapMemoryMB * k8sOverheadFactor.get.toDouble).toLong
              } else {
                // For JVM-based jobs this value will default to 0.10 and 0.40 for non-JVM jobs
                // TODO - We can't tell above by any property... do we try to parse submit cli?
                (execHeapMemoryMB * 0.1).toLong
              }
            } else if (sparkMaster.startsWith("spark:")) {
              // would be the entire node memory by default, we don't know and user doesn't
              // need to specify
              0L
            } else {
              0L
            }
        }
      }
    } else {
      0L
    }
  }

  def createClusterInfo(coresPerExecutor: Int,
      numExecsPerNode: Int,
      numExecs: Int,
      numWorkerNodes: Int,
      sparkProperties: Map[String, String],
      systemProperties: Map[String, String]): SourceClusterInfo = {
    val driverHost = sparkProperties.get("spark.driver.host")
    val executorHeapMem = getExecutorHeapMemoryMB(sparkProperties.get)
    val dynamicAllocSettings = Platform.getDynamicAllocationSettings(sparkProperties)
    SourceClusterInfo(platformName, coresPerExecutor, numExecsPerNode, numExecs, numWorkerNodes,
      executorHeapMem, dynamicAllocSettings.enabled, dynamicAllocSettings.max,
      dynamicAllocSettings.min, dynamicAllocSettings.initial, driverHost = driverHost)
  }

  // set the cluster information for this platform based on what we found in the
  // eventlog
  def configureClusterInfoFromEventLog(coresPerExecutor: Int,
      execsPerNode: Int,
      numExecs: Int,
      numExecutorNodes: Int,
      sparkProperties: Map[String, String],
      systemProperties: Map[String, String]): Unit = {
    clusterInfoFromEventLog = Some(createClusterInfo(coresPerExecutor, execsPerNode,
      numExecs, numExecutorNodes, sparkProperties, systemProperties))
  }

  /**
   * Indicate if the platform is a cloud service provider.
   */
  def isPlatformCSP: Boolean = false

  /**
   * Indicate if the platform requires path recommendations
   */
  def requirePathRecommendations: Boolean = true

  /**
   * The maximum number of GPUs supported by any instance type in this platform.
   */
  lazy val maxGpusSupported: Int = {
    val gpuCounts = getInstanceMapByName.values.map(_.numGpus)
    if (gpuCounts.isEmpty) {
      // Default to 1 if instance types are not defined (e.g. on-prem)
      1
    } else {
      gpuCounts.max
    }
  }

  /**
   * Get the mapping of instance names to their corresponding instance information.
   */
  def getInstanceMapByName: Map[NodeInstanceMapKey, InstanceInfo] = Map.empty

  /**
   * Creates the recommended GPU cluster info based on Spark properties and
   * cluster properties (either provided explicitly by the user or
   * inferred from the event log).
   *
   * @param sourceSparkProperties A map of Spark properties (combined from application and
   *                        cluster properties)
   * @return Optional `RecommendedClusterInfo` containing the GPU cluster configuration
   *         recommendation.
   */
  def createRecommendedGpuClusterInfo(
      recommendations: mutable.LinkedHashMap[String, TuningEntryTrait],
      sourceSparkProperties: Map[String, String],
      recommendedClusterSizingStrategy: ClusterSizingStrategy): Unit = {
    // Get the appropriate cluster configuration strategy (either
    // 'ClusterPropertyBasedStrategy' based on cluster properties or
    // 'EventLogBasedStrategy' based on the event log).
    val configurationStrategyOpt = ClusterConfigurationStrategy.getStrategy(
      platform = this,
      recommendations = recommendations,
      sourceSparkProperties = sourceSparkProperties,
      recommendedClusterSizingStrategy = recommendedClusterSizingStrategy)

    configurationStrategyOpt match {
      case Some(strategy) =>
        // Using the strategy, generate the recommended cluster configuration (num executors,
        // cores per executor, num executors per node).
        strategy.getRecommendedConfig match {
          case Some(clusterConfig) =>
            // If recommended node instance information is not already set, create it based on the
            // existing cluster configuration and platform defaults.
            val _recommendedWorkerNode = recommendedWorkerNode.getOrElse {
              InstanceInfo.createDefaultInstance(
                cores = clusterConfig.coresPerNode,
                memoryMB = clusterConfig.memoryPerNodeMb,
                numGpus = clusterConfig.numGpusPerNode,
                gpuDevice = clusterConfig.gpuDevice)
            }
            val vendor = clusterInfoFromEventLog.map(_.vendor).getOrElse("")

            // Determine the driver node type in the following order:
            // 1. Use the value specified in the target cluster, if present
            // 2. Else, if it was inferred from the event log
            //
            // Notes:
            // - If the driver node type is still undefined and the user is running
            //   the Qualification Tool, a platform default will be set later.
            // - Using a platform default for the driver node type is not applicable
            //   when running the Profiling Tool.
            val recommendedDriverNode = targetCluster.map(_.getDriverInfo.getInstanceType)
              .filter(_.nonEmpty)
              .orElse(clusterInfoFromEventLog.flatMap(_.driverNodeType))

            val numWorkerNodes = if (!isPlatformCSP) {
              // For on-prem, we do not have the concept of worker nodes.
              -1
            } else {
              // Calculate number of worker nodes based on executors and GPUs per instance
              math.ceil(clusterConfig.numExecutors.toDouble /
                _recommendedWorkerNode.numGpus).toInt
            }

            // Calculate cores per executor: use the user-specified value (enforced, or
            // preserved from the source application) if valid, otherwise divide total
            // cores in the instance by the number of GPUs.
            val defaultCoresPerExecutor = math.ceil(
              _recommendedWorkerNode.cores.toDouble / _recommendedWorkerNode.numGpus
            ).toInt
            val recommendedCoresPerExecutor =
              getEnforcedOrPreservedSparkProperty("spark.executor.cores", sourceSparkProperties.get)
              .flatMap(v => scala.util.Try(v.toInt).toOption)
              .filter { specifiedCores =>
                val maxCores = _recommendedWorkerNode.cores
                val isValid = specifiedCores > 0 && specifiedCores <= maxCores
                if (!isValid) {
                  logWarning(s"User-specified spark.executor.cores=$specifiedCores is invalid " +
                    s"(must be > 0 and <= $maxCores cores on instance " +
                    s"${_recommendedWorkerNode.name}). " +
                    s"Using default value: $defaultCoresPerExecutor")
                }
                isValid
              }
              .getOrElse(defaultCoresPerExecutor)

            // Calculate the recommended number of executors by dividing the total number of
            // executors by the recommended cores per executor
            val recommendedNumExecutors =
              (clusterConfig.coresPerExec * clusterConfig.numExecutors) /
                recommendedCoresPerExecutor

            val dynamicAllocSettings = Platform.getDynamicAllocationSettings(sourceSparkProperties)
            recommendedWorkerNode = Some(_recommendedWorkerNode)
            recommendedClusterInfo = Some(RecommendedClusterInfo(
              vendor = vendor,
              coresPerExecutor = recommendedCoresPerExecutor,
              numWorkerNodes = numWorkerNodes,
              numGpusPerNode = _recommendedWorkerNode.numGpus,
              numExecutors = recommendedNumExecutors,
              gpuDevice = getPlatformGpuDeviceName(_recommendedWorkerNode.gpuDevice),
              dynamicAllocationEnabled = dynamicAllocSettings.enabled,
              dynamicAllocationMaxExecutors = dynamicAllocSettings.max,
              dynamicAllocationMinExecutors = dynamicAllocSettings.min,
              dynamicAllocationInitialExecutors = dynamicAllocSettings.initial,
              driverNodeType = recommendedDriverNode,
              workerNodeType = Some(_recommendedWorkerNode.name)
            ))

          case None =>
            logWarning("Failed to generate a cluster recommendation. " +
              "Could not determine number of executors. " +
              "Check the Spark properties used for this application or " +
              "cluster properties (if provided).")
        }

      case None =>
        logWarning("Failed to generate a cluster recommendation. " +
          "Could not determine number of executors. " +
          "Cluster properties are missing and event log does not contain cluster information.")
    }
  }

  /**
   * Get the user-enforced Spark property for the given property key.
   */
  final def getUserEnforcedSparkProperty(propertyKey: String): Option[String] = {
    userEnforcedRecommendations.get(propertyKey)
  }

  /**
   * Resolve the baseline value to use for a Spark property during recommendation
   * calculations. Precedence: user-enforced value first, then (if the property is
   * in the target cluster's `preserve` list) its source-application value. Returns
   * None if neither applies (caller falls back to its computed default).
   *
   * @param propertyKey  the Spark property name
   * @param sourceLookup accessor for the source-application property value
   */
  final def getEnforcedOrPreservedSparkProperty(
      propertyKey: String,
      sourceLookup: String => Option[String]): Option[String] = {
    getUserEnforcedSparkProperty(propertyKey)
      .orElse(if (isPropertyPreserved(propertyKey)) sourceLookup(propertyKey) else None)
  }

  /**
   * Check if the property is preserved in the target cluster.
   */
  final def isPropertyPreserved(propertyKey: String): Boolean = {
    targetCluster.exists(_.getSparkProperties.preservePropertiesSet.contains(propertyKey))
  }

  /**
   * Check if the property is excluded in the target cluster.
   */
  final def isPropertyExcluded(propertyKey: String): Boolean = {
    targetCluster.exists(_.getSparkProperties.excludePropertiesSet.contains(propertyKey))
  }

  /**
   * Check if the property has a user override from the target cluster — either an
   * `enforced` value or a `preserve` directive.
   */
  final def isPropertyUserOverridden(propertyKey: String): Boolean = {
    getUserEnforcedSparkProperty(propertyKey).isDefined || isPropertyPreserved(propertyKey)
  }

  /**
   * Set the default driver node type for the recommended cluster if it is missing.
   */
  final def setDefaultDriverNodeToRecommendedClusterIfMissing() : Unit = {
    recommendedClusterInfo = recommendedClusterInfo.map { recClusterInfo =>
      if (recClusterInfo.driverNodeType.isEmpty) {
        recClusterInfo.copy(driverNodeType = defaultRecommendedDriverNode.map(_.instanceType))
      } else {
        recClusterInfo
      }
    }
  }
}

abstract class DatabricksPlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends Platform(gpuDevice, targetCluster) {
  override val sparkVersionLabel: String = "Databricks runtime"
  override def isPlatformCSP: Boolean = true

  override val supportedRuntimes: Set[SparkRuntime.SparkRuntime] = Set(
    SparkRuntime.SPARK, SparkRuntime.SPARK_RAPIDS, SparkRuntime.PHOTON
  )

  // note that Databricks generally sets the spark.executor.memory for the user.  Our
  // auto tuner heuristics generally sets it lower then Databricks so go ahead and
  // allow our auto tuner to take affect for this in anticipation that we will use more
  // off heap memory.
  override val recommendationsToExclude: Set[String] = Set(
    "spark.executor.cores",
    "spark.executor.instances",
    "spark.executor.memoryOverhead"
  )

  // Supported Databricks version to RapidsShuffleManager version mapping.
  // TODO: Issue to automate this https://github.com/NVIDIA/cudf-spark-tools/issues/1676
  override val supportedShuffleManagerVersionMap: Array[(String, String)] = Array(
    "12.2" -> "332db",
    "13.3" -> "341db"
  )

  override def createClusterInfo(coresPerExecutor: Int,
      numExecsPerNode: Int,
      numExecs: Int,
      numWorkerNodes: Int,
      sparkProperties: Map[String, String],
      systemProperties: Map[String, String]): SourceClusterInfo = {
    val workerNodeType = sparkProperties.get(DatabricksParseHelper.PROP_WORKER_TYPE_ID_KEY)
    val driverNodeType = sparkProperties.get(DatabricksParseHelper.PROP_DRIVER_TYPE_ID_KEY)
    val clusterId = sparkProperties.get(DatabricksParseHelper.PROP_TAG_CLUSTER_ID_KEY)
    val driverHost = sparkProperties.get("spark.driver.host")
    val clusterName = sparkProperties.get(DatabricksParseHelper.PROP_TAG_CLUSTER_NAME_KEY)
    val executorHeapMem = getExecutorHeapMemoryMB(sparkProperties.get)
    val dynamicAllocSettings = Platform.getDynamicAllocationSettings(sparkProperties)
    SourceClusterInfo(platformName, coresPerExecutor, numExecsPerNode, numExecs, numWorkerNodes,
      executorHeapMem, dynamicAllocSettings.enabled, dynamicAllocSettings.max,
      dynamicAllocSettings.min, dynamicAllocSettings.initial, driverNodeType,
      workerNodeType, driverHost, clusterId, clusterName)
  }
}

class DatabricksAwsPlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends DatabricksPlatform(gpuDevice, targetCluster)
  with Logging {
  override def platformName: String = PlatformNames.DATABRICKS_AWS

  override def getInstanceMapByName: Map[NodeInstanceMapKey, InstanceInfo] = {
    PlatformInstanceTypes.DATABRICKS_AWS_BY_INSTANCE_NAME
  }

  /**
   * Could not find public documentation for this. This was determined based on
   * manual inspection of Databricks AWS configurations. 35% reserved.
   */
  override def nonExecutorMemoryFraction = 0.35
}

class DatabricksAzurePlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends DatabricksPlatform(gpuDevice, targetCluster) {
  override def platformName: String = PlatformNames.DATABRICKS_AZURE

  override def getInstanceMapByName: Map[NodeInstanceMapKey, InstanceInfo] = {
    PlatformInstanceTypes.AZURE_NCAS_T4_V3_BY_INSTANCE_NAME
  }

  /**
   * Could not find public documentation for this. This was determined based on
   * manual inspection of Databricks Azure configurations. 30% reserved.
   */
  override def nonExecutorMemoryFraction = 0.3
}

class DataprocPlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends Platform(gpuDevice, targetCluster) {
  override def platformName: String = PlatformNames.DATAPROC

  // scalastyle:off line.size.limit
  /**
   * For YARN on Dataproc, 20% of memory is reserved by the resource manager.
   * yarn.nodemanager.resource.memory-mb is typically set to 80% of total memory.
   * Reference: https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/autoscaling#hadoop_yarn_metrics
   */
  // scalastyle:on line.size.limit
  override def nonExecutorMemoryFraction: Double = 0.2

  override val platformSpecificRecommendations: Map[String, String] = Map(
    // Keep disabled. This property does not work well with GPU clusters.
    "spark.dataproc.enhanced.optimizer.enabled" -> "false",
    // Keep disabled. This property does not work well with GPU clusters.
    "spark.dataproc.enhanced.execution.enabled" -> "false"
  )

  override def isPlatformCSP: Boolean = true

  override def getInstanceMapByName: Map[NodeInstanceMapKey, InstanceInfo] = {
    PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME
  }

  /**
   * Get the platform-specific GPU device name for Dataproc.
   * Returns GPU device names in the format expected by Dataproc.
   * Reference: https://docs.cloud.google.com/dataproc/docs/concepts/compute/gpus#types_of_gpus
   *
   * Supported GPU devices mapped to GCP accelerator type names:
   * - L4 -> nvidia-l4
   * - A100 -> nvidia-a100-80gb
   * - P100 -> nvidia-tesla-p100
   * - V100 -> nvidia-tesla-v100
   * - P4 -> nvidia-tesla-p4
   * - T4 -> nvidia-tesla-t4
   *
   * @param gpuDevice The GPU device to convert to Dataproc-specific name
   * @return Dataproc-specific GPU device name
   */
  override def getPlatformGpuDeviceName(gpuDevice: GpuDevice): String = {
    gpuDevice match {
      case L4Gpu => "nvidia-l4"
      case A100Gpu => "nvidia-a100-80gb"
      case P100Gpu => "nvidia-tesla-p100"
      case V100Gpu => "nvidia-tesla-v100"
      case P4Gpu => "nvidia-tesla-p4"
      case T4Gpu => "nvidia-tesla-t4"
      case _ => gpuDevice.toString  // Fallback to default for unsupported devices
    }
  }
}

class DataprocServerlessPlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends DataprocPlatform(gpuDevice, targetCluster) {

  override def platformName: String = PlatformNames.DATAPROC_SL
  override def defaultGpuDevice: GpuDevice = L4Gpu
  override def isPlatformCSP: Boolean = true
}

class DataprocGkePlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends DataprocPlatform(gpuDevice, targetCluster) {

  override def platformName: String = PlatformNames.DATAPROC_GKE
  override def isPlatformCSP: Boolean = true
}

class EmrPlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends Platform(gpuDevice, targetCluster) {
  override def platformName: String = PlatformNames.EMR

  // EMR needs to track extraJavaOptions to disable THP
  override val platformEnabledTuningEntries: Set[String] = Set(
    "spark.driver.extraJavaOptions",
    "spark.executor.extraJavaOptions"
  )

  // scalastyle:off line.size.limit
  /**
   * For YARN on EMR, 30% of memory is reserved by the resource manager.
   * yarn.nodemanager.resource.memory-mb is typically set to 70% of total memory.
   * Reference: https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-hadoop-task-config.html#emr-hadoop-task-config-g6
   */
  // scalastyle:on line.size.limit
  override def nonExecutorMemoryFraction: Double = 0.3

  override def isPlatformCSP: Boolean = true
  override def requirePathRecommendations: Boolean = false

  override def getRetainedSystemProps: Set[String] = Set("EMR_CLUSTER_ID", "EMR_STEP_ID")

  override def createClusterInfo(coresPerExecutor: Int,
      numExecsPerNode: Int,
      numExecs: Int,
      numWorkerNodes: Int,
      sparkProperties: Map[String, String],
      systemProperties: Map[String, String]): SourceClusterInfo = {
    val clusterId = systemProperties.get("EMR_CLUSTER_ID")
    val driverHost = sparkProperties.get("spark.driver.host")
    val executorHeapMem = getExecutorHeapMemoryMB(sparkProperties.get)
    val dynamicAllocSettings = Platform.getDynamicAllocationSettings(sparkProperties)
    SourceClusterInfo(platformName, coresPerExecutor, numExecsPerNode, numExecs,
      numWorkerNodes, executorHeapMem, dynamicAllocSettings.enabled, dynamicAllocSettings.max,
      dynamicAllocSettings.min, dynamicAllocSettings.initial, clusterId = clusterId,
      driverHost = driverHost)
  }

  override def getInstanceMapByName: Map[NodeInstanceMapKey, InstanceInfo] = {
    PlatformInstanceTypes.EMR_BY_INSTANCE_NAME
  }
}

class OnPremPlatform(gpuDevice: Option[GpuDevice],
    targetCluster: Option[TargetClusterProps])
  extends Platform(gpuDevice, targetCluster) {

  override def platformName: String = PlatformNames.ONPREM
  override def defaultGpuDevice: GpuDevice = L4Gpu
  override val defaultGpuForSpeedupFactor: GpuDevice = A100Gpu
  // on prem is hard since we don't know what node configurations they have
  // assume 1 for now. We should have them pass this information in the future.
  override lazy val maxGpusSupported: Int = 1

  /**
   * For OnPrem, no memory is reserved by resource manager since we are calculating the
   * overall memory by ourselves or it is provided by the user.
   *
   * See `getMemoryPerNodeMb()` in [[com.nvidia.spark.rapids.tool.ClusterConfigurationStrategy]]
   */
  def nonExecutorMemoryFraction: Double = 0.0

  /**
   * Returns the recommended instance info based on the provided WorkerInfo for OnPrem.
   *
   * Behavior:
   * - Create an InstanceInfo with the specified CPU cores, memory, and GPU.
   * - If the workerInfo or the gpu info is absent, skips creating recommended instance type.
   */
  override def getRecommendedInstanceInfoFromTargetWorker(
      workerInfoOpt: Option[WorkerInfo]): Option[InstanceInfo] = {
    workerInfoOpt.flatMap { workerInfo =>
      workerInfo.getGpu.device.map { gpuDevice =>
        InstanceInfo.createDefaultInstance(
          cores = workerInfo.cpuCores,
          memoryMB = workerInfo.memoryGB * 1024L,
          numGpus = workerInfo.getGpu.count,
          gpuDevice = gpuDevice)
      }
    }.orElse {
      logDebug("Worker info or Gpu info is not provided in the target cluster. " +
        "Skipping recommended instance info creation.")
      None
    }
  }
}

object Platform {
  def isDynamicAllocationEnabled(sparkProperties: Map[String, String]): Boolean = {
    sparkProperties.getOrElse("spark.dynamicAllocation.enabled", "false").toBoolean
  }

  def getDynamicAllocationSettings(sparkProperties: Map[String, String]): DynamicAllocationInfo = {
    val dynamicAllocationEnabled = isDynamicAllocationEnabled(sparkProperties)
    if (dynamicAllocationEnabled) {
      val dynamicAllocationMax = sparkProperties.
        getOrElse("spark.dynamicAllocation.maxExecutors", Int.MaxValue.toString)
      val dynamicAllocationMin = sparkProperties.
        getOrElse("spark.dynamicAllocation.minExecutors", "0")
      val dynamicAllocationInit = sparkProperties.
        getOrElse("spark.dynamicAllocation.initialExecutors", sparkProperties.
          getOrElse("spark.executor.instances", dynamicAllocationMin))
      DynamicAllocationInfo(dynamicAllocationEnabled, dynamicAllocationMax,
        dynamicAllocationMin, dynamicAllocationInit)
    } else {
      DynamicAllocationInfo(dynamicAllocationEnabled, "N/A", "N/A", "N/A")
    }
  }
}

/**
 * Factory for creating instances of different platforms.
 * This factory supports various platforms and provides methods for creating
 * corresponding platform instances.
 */
object PlatformFactory extends Logging {
  /**
   * Extracts platform and GPU names from the provided platform key.
   * Assumption: If the last part contains a number, we assume it is GPU name
   *
   * E.g.,
   * - 'emr-t4': Platform: emr, GPU: t4
   * - 'dataproc-gke-l4': Platform dataproc-gke, GPU: l4
   * - 'databricks-aws': Platform databricks-aws, GPU: None
   */
  private def extractPlatformGpuName(platformKey: String): (String, Option[String]) = {
    val parts = platformKey.split('-')
    val numberPattern = ".*\\d.*".r
    // If the last part contains a number, we assume it is GPU name
    if (numberPattern.findFirstIn(parts.last).isDefined) {
      (parts.init.toList.mkString("-"), Some(parts.last))
    } else {
      // If no GPU information is present, return the entire platform key as the
      // platform name and None for GPU
      (parts.toList.mkString("-"), None)
    }
  }

  @throws[IllegalArgumentException]
  @tailrec
  private def createPlatformInstance(platformName: String,
      gpuDevice: Option[GpuDevice],
      targetCluster: Option[TargetClusterProps]): Platform = platformName match {
    case PlatformNames.DATABRICKS_AWS =>
      new DatabricksAwsPlatform(gpuDevice, targetCluster)
    case PlatformNames.DATABRICKS_AZURE =>
      new DatabricksAzurePlatform(gpuDevice, targetCluster)
    case PlatformNames.DATAPROC =>
      new DataprocPlatform(gpuDevice, targetCluster)
    case PlatformNames.DATAPROC_GKE =>
      new DataprocGkePlatform(gpuDevice, targetCluster)
    case PlatformNames.DATAPROC_SL =>
      new DataprocServerlessPlatform(gpuDevice, targetCluster)
    case PlatformNames.EMR =>
      new EmrPlatform(gpuDevice, targetCluster)
    case PlatformNames.ONPREM =>
      new OnPremPlatform(gpuDevice, targetCluster)
    case p if p.isEmpty =>
      logInfo(s"Platform is not specified. Using ${PlatformNames.DEFAULT} as default.")
      createPlatformInstance(PlatformNames.DEFAULT, gpuDevice, targetCluster)
    case _ =>
      throw new IllegalArgumentException(s"Unsupported platform: $platformName. " +
        s"Options include ${PlatformNames.getAllNames.mkString(", ")}.")
  }

  /**
   * Creates an instance of `Platform` based on the specified platform key.
   *
   * @param platformKey The key identifying the platform. Defaults to `PlatformNames.DEFAULT`.
   * @param targetClusterProps Optional target cluster properties if the user specified them.
   */
  def createInstance(platformKey: String = PlatformNames.DEFAULT,
      targetClusterProps: Option[TargetClusterProps] = None): Platform = {
    // TODO: Remove platformKey containing GPU name
    val (platformName, gpuName) = extractPlatformGpuName(platformKey)
    if (gpuName.nonEmpty) {
      logWarning("Using GPU name in platform key will be deprecated in future. " +
        "Please use the '--target-worker-info' option to specify the GPU type.")
    }
    val gpuDevice = gpuName.flatMap(GpuDevice.createInstance)
    // case when gpu name is detected but not in device map
    if (gpuName.isDefined && gpuDevice.isEmpty) {
      throw new IllegalArgumentException(s"Unsupported GPU device: ${gpuName.get}. " +
          s"Supported GPU devices are: ${GpuDevice.deviceMap.keys.mkString(", ")}.")
    }
    val platform = createPlatformInstance(platformName, gpuDevice,
      targetClusterProps)
    logDebug(s"Using platform: $platform")
    platform
  }
}
