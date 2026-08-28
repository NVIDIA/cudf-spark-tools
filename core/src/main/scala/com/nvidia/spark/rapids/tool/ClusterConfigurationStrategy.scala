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

package com.nvidia.spark.rapids.tool

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.tuning.{AutoTuner, TuningEntryTrait}

import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.SourceClusterInfo
import org.apache.spark.sql.rapids.tool.util.StringUtils

/**
 * Configuration for the recommended cluster to run the application with the cuDF plugin.
 */
case class RecommendedClusterConfig(
    numExecutors: Int,
    coresPerExec: Int,
    memoryPerNodeMb: Long, // For onprem or cases where a matching CSP instance type is unavailable
    gpuDevice: GpuDevice,
    numGpusPerNode: Int
) {
  def execsPerNode: Int = {
    numGpusPerNode
  }

  def coresPerNode: Int = {
    coresPerExec * execsPerNode
  }
}

/**
 * Base strategy trait for determining the sizing of the cluster configuration.
 */
trait ClusterSizingStrategy {

  /**
   * Utility method to compute recommended cores per executor.
   *
   * Precedence: user-enforced spark.executor.cores, then (if preserved) the
   * source value, then the platform default. Non-numeric / non-positive values
   * fall through to the default. `sourceLookup` reads the source-application
   * properties (NOT event-log-derived initial values).
   *
   * TODO(#2053): only non-positive values are rejected here; a value exceeding the
   * worker's core count is validated separately in Platform, so the two can diverge
   * on invalid input. Centralizing the validation is a deferred follow-up.
   */
  final def computeRecommendedCoresPerExec(
      platform: Platform,
      totalCoresCount: Int,
      sourceLookup: String => Option[String]): Int = {
    platform.getEnforcedOrPreservedSparkProperty("spark.executor.cores", sourceLookup)
      .flatMap(v => scala.util.Try(v.toInt).toOption)
      .filter(_ > 0)
      .getOrElse {
        if (platform.isPlatformCSP) {
          // For CSPs, we already have the recommended cores per executor based on the instance type
          platform.recommendedCoresPerExec
        } else {
          // For onprem, we do want to limit to the total cores count
          // Sanity check to avoid returning 0 cores per executor in case
          // set wrongly by the user
          if (totalCoresCount > 0) {
            math.min(platform.recommendedCoresPerExec, totalCoresCount)
          } else {
            platform.recommendedCoresPerExec
          }
        }
      }
  }

  final def computeRecommendedInstances(
      platform: Platform,
      sourceLookup: String => Option[String],
      computeRecommendedInstancesExpr: () => Int): Int = {
    platform.getEnforcedOrPreservedSparkProperty("spark.executor.instances", sourceLookup)
      .flatMap(v => scala.util.Try(v.toInt).toOption)
      .filter(_ > 0)
      .getOrElse(computeRecommendedInstancesExpr())
  }

  /** Abstract method to compute the recommended cluster configuration. */
  def computeRecommendedConfig(
    platform: Platform,
    initialNumExecutors: Int,
    initialCoresPerExec: Int,
    sourceSparkProperties: Map[String, String],
    getMemoryPerNodeMb: => Long,
    getRecommendedGpuDevice: => GpuDevice,
    getRecommendedNumGpus: => Int): RecommendedClusterConfig
}

/**
 * Strategy that keeps the total number of CPU cores between the source and
 * target clusters constant. It adjusts the number of executors (and hence GPUs)
 * accordingly.
 */
object ConstantTotalCoresStrategy extends ClusterSizingStrategy {
  def computeRecommendedConfig(
      platform: Platform,
      initialNumExecutors: Int,
      initialCoresPerExec: Int,
      sourceSparkProperties: Map[String, String],
      getMemoryPerNodeMb: => Long,
      getRecommendedGpuDevice: => GpuDevice,
      getRecommendedNumGpus: => Int): RecommendedClusterConfig = {
    val totalCoresCount = initialCoresPerExec * initialNumExecutors
    val recommendedCoresPerExec =
      computeRecommendedCoresPerExec(platform, totalCoresCount, sourceSparkProperties.get)
    val recommendedNumExecutors = computeRecommendedInstances(platform, sourceSparkProperties.get,
      () => math.ceil(totalCoresCount.toDouble / recommendedCoresPerExec).toInt)
    RecommendedClusterConfig(recommendedNumExecutors, recommendedCoresPerExec,
      getMemoryPerNodeMb, getRecommendedGpuDevice, getRecommendedNumGpus)
  }
}

/**
 * Strategy that keeps the total number of GPUs between the source and
 * target clusters constant. The number of executors remains unchanged.
 */
object ConstantGpuCountStrategy extends ClusterSizingStrategy {
  def computeRecommendedConfig(
      platform: Platform,
      initialNumExecutors: Int,
      initialCoresPerExec: Int,
      sourceSparkProperties: Map[String, String],
      getMemoryPerNodeMb: => Long,
      getRecommendedGpuDevice: => GpuDevice,
      getRecommendedNumGpus: => Int): RecommendedClusterConfig = {
    val totalCoresCount = initialCoresPerExec * initialNumExecutors
    val recommendedCoresPerExec =
      computeRecommendedCoresPerExec(platform, totalCoresCount, sourceSparkProperties.get)
    val recommendNumExecutors = computeRecommendedInstances(platform, sourceSparkProperties.get,
      () => initialNumExecutors)
    RecommendedClusterConfig(recommendNumExecutors, recommendedCoresPerExec,
      getMemoryPerNodeMb, getRecommendedGpuDevice, getRecommendedNumGpus)
  }
}

/**
 * Strategy that preserves the source cluster's cores-per-executor and executor count.
 * Used in profiling mode on fixed (on-prem) hardware where node slicing is not
 * changeable. User-enforced overrides are still honored.
 */
object SourceCoresPreservingStrategy extends ClusterSizingStrategy {
  def computeRecommendedConfig(
      platform: Platform,
      initialNumExecutors: Int,
      initialCoresPerExec: Int,
      sourceSparkProperties: Map[String, String],
      getMemoryPerNodeMb: => Long,
      getRecommendedGpuDevice: => GpuDevice,
      getRecommendedNumGpus: => Int): RecommendedClusterConfig = {
    val coresPerExec =
      platform.getEnforcedOrPreservedSparkProperty(
          "spark.executor.cores", sourceSparkProperties.get)
        .flatMap(v => scala.util.Try(v.toInt).toOption)
        .filter(_ > 0)
        .getOrElse(initialCoresPerExec)
    val numExecutors =
      computeRecommendedInstances(platform, sourceSparkProperties.get, () => initialNumExecutors)
    RecommendedClusterConfig(numExecutors, coresPerExec,
      getMemoryPerNodeMb, getRecommendedGpuDevice, getRecommendedNumGpus)
  }
}

/**
 * Base trait for different cluster configuration strategies.
 */
abstract class ClusterConfigurationStrategy(
    platform: Platform,
    sourceSparkProperties: Map[String, String],
    recommendedClusterSizingStrategy: ClusterSizingStrategy) {

  /**
   * Calculates the initial number of executors based on the strategy.
   */
  protected def calculateInitialNumExecutors: Int

  private def getInitialNumExecutors: Int = {
    val dynamicAllocationEnabled = Platform.isDynamicAllocationEnabled(sourceSparkProperties)
    val execInstFromProps = sourceSparkProperties.get("spark.executor.instances")
    // If dynamic allocation is disabled, use spark.executor.instances in precedence
    if (execInstFromProps.isDefined && !dynamicAllocationEnabled) {
      // Spark Properties are in order:
      // P0. User defined software properties in Cluster Properties
      // P1. Spark Properties defined in the application information
      execInstFromProps.get.toInt
    } else {
      calculateInitialNumExecutors
    }
  }

  /**
   * Calculates the initial number of cores per executor based on the strategy.
   */
  protected def calculateInitialCoresPerExec: Int

  private def getInitialCoresPerExec: Int = {
    val coresFromProps = sourceSparkProperties.get("spark.executor.cores")
    // Use spark.executor.cores in precedence
    if (coresFromProps.isDefined) {
      coresFromProps.get.toInt
    } else {
      calculateInitialCoresPerExec
    }
  }

  protected def getRecommendedMemoryPerNodeMb: Long

  protected def getSourceNumGpus: Option[Int]

  protected def getSourceGpuDevice: Option[GpuDevice]

  protected def getRecommendedNumGpus: Int

  protected def getRecommendedGpuDevice: GpuDevice

  /**
   * Generates the recommended cluster configuration based on the strategy.
   *
   * Logic:
   * 1. Calculate the initial number of executors and cores per executor.
   * 2. Calculate the total core count by multiplying initial cores per executor
   *    by the initial number of executors.
   * 3. Retrieve the recommended cores per executor from the platform (default is 16),
   *    for onprem, limit the recommended cores per executor to the total core count.
   * 4. Calculate the recommended number of executors by dividing the total core count
   *    by the recommended cores per executor.
   */
  final def getRecommendedConfig: Option[RecommendedClusterConfig] = {
    // The sanity check for number of executors is only added because
    // we do not have a default alternative for it
    val initialNumExecutors = getInitialNumExecutors
    if (initialNumExecutors <= 0) {
      None
    } else {
      Some(recommendedClusterSizingStrategy.computeRecommendedConfig(
        platform,
        initialNumExecutors,
        getInitialCoresPerExec,
        sourceSparkProperties,
        getRecommendedMemoryPerNodeMb,
        getRecommendedGpuDevice,
        getRecommendedNumGpus
      ))
    }
  }
}

/**
 * Strategy for cluster configuration based on cluster information from event log.
 */
class EventLogBasedStrategy(
    platform: Platform,
    recommendations: mutable.LinkedHashMap[String, TuningEntryTrait],
    sourceSparkProperties: Map[String, String],
    recommendedClusterSizingStrategy: ClusterSizingStrategy)
  extends ClusterConfigurationStrategy(platform, sourceSparkProperties,
    recommendedClusterSizingStrategy) {

  private val clusterInfoFromEventLog: SourceClusterInfo = {
    platform.clusterInfoFromEventLog.getOrElse(
      throw new IllegalArgumentException("Cluster information from event log must be defined"))
  }

  // scalastyle:off line.size.limit
  /**
   * For onprem or cases where a matching CSP instance type is unavailable,
   * this method returns the memory for the recommended node in MB.
   *
   * Reference:
   * https://spark.apache.org/docs/3.5.5/configuration.html#:~:text=spark.executor.memoryOverhead,pyspark.memory.
   */
  // scalastyle:on line.size.limit
  override def getRecommendedMemoryPerNodeMb: Long = {
    val heapMemMB = platform.getEnforcedOrPreservedSparkProperty("spark.executor.memory",
        sourceSparkProperties.get)
      .map(StringUtils.convertToMB(_, Some(ByteUnit.BYTE)))
      .getOrElse(clusterInfoFromEventLog.executorHeapMemory)
    // Get a combined spark properties function that includes user enforced properties
    // and properties from the event log
    val sparkPropertiesFn = AutoTuner.getCombinedPropertyFn(recommendations,
      sourceSparkProperties)
    val overheadMemMB = platform.getExecutorOverheadMemoryMB(sparkPropertiesFn)
    val sparkOffHeapMemMB = platform.getSparkOffHeapMemoryMB(sparkPropertiesFn)
      .getOrElse(0L)
    val pySparkMemMB = platform.getPySparkMemoryMB(sparkPropertiesFn).getOrElse(0L)
    heapMemMB + overheadMemMB + sparkOffHeapMemMB + pySparkMemMB
  }

  override def calculateInitialNumExecutors: Int = {
    clusterInfoFromEventLog.numExecutors
  }

  override def calculateInitialCoresPerExec: Int = {
    clusterInfoFromEventLog.coresPerExecutor
  }

  // TODO: Extract the GPU device on the source cluster node
  def getSourceGpuDevice: Option[GpuDevice] = {
    None
  }

  // TODO: Extract the GPU count on the source cluster node
  def getSourceNumGpus: Option[Int] = {
    None
  }

  def getRecommendedGpuDevice: GpuDevice = {
    platform.recommendedGpuDevice
  }

  def getRecommendedNumGpus: Int = {
    platform.recommendedNumGpus
  }
}

/**
 * Companion object to create appropriate cluster configuration strategy.
 *
 * Uses event log based strategy when cluster information from event log is available.
 */
object ClusterConfigurationStrategy {
  def getStrategy(
      platform: Platform,
      recommendations: mutable.LinkedHashMap[String, TuningEntryTrait],
      sourceSparkProperties: Map[String, String],
      recommendedClusterSizingStrategy: ClusterSizingStrategy)
  : Option[ClusterConfigurationStrategy] = {
    if (platform.clusterInfoFromEventLog.isDefined) {
      // Use strategy based on cluster information from event log
      Some(new EventLogBasedStrategy(platform, recommendations,
        sourceSparkProperties, recommendedClusterSizingStrategy))
    } else {
      // Cluster information from event log is not available
      None
    }
  }
}
