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

package com.nvidia.spark.rapids.tool.tuning

import java.util

import scala.beans.BeanProperty
import scala.jdk.CollectionConverters._

import com.nvidia.spark.rapids.tool.NodeInstanceMapKey
import com.nvidia.spark.rapids.tool.tuning.config.TuningEntryDefinition

import org.apache.spark.sql.rapids.tool.util.ValidatableProperties

/**
 * Base class to hold the worker information for the target cluster
 * See subclasses below for concrete implementations.
 *
 * @param instanceType The instance type (e.g., "g2-standard-24" for Dataproc)
 * @param cpuCores Number of CPU cores (for OnPrem configurations)
 * @param memoryGB Total memory in GB (for OnPrem configurations)
 * @param gpu GPU configuration
 */
class WorkerInfo (
  @BeanProperty var instanceType: String,
  @BeanProperty var cpuCores: Int,
  @BeanProperty var memoryGB: Long,
  @BeanProperty var gpu: GpuWorkerProps) extends ValidatableProperties {

  def this() = this("", 0, 0L, new GpuWorkerProps())

  /** Returns true if all resource fields are unset or empty. */
  def isEmpty: Boolean = {
    (instanceType == null || instanceType.isEmpty) &&
      (cpuCores == 0 && memoryGB == 0L) && gpu.isEmpty
  }

  /** Returns true if this represents a CSP Info (instance type is set). */
  def isCspInfo: Boolean = {
    instanceType != null && instanceType.nonEmpty
  }

  /** Returns true if this represents an OnPrem Info (CPU or memory is set). */
  def isOnpremInfo: Boolean = {
    cpuCores > 0 || memoryGB > 0
  }

  override def validate(): Unit = {
    if (!isEmpty && isCspInfo && isOnpremInfo) {
      throw new IllegalArgumentException(
        "Both instance type and OnPrem resource values are provided. Please specify either one")
    }
    if (isOnpremInfo && gpu.isEmpty) {
      throw new IllegalArgumentException("GPU information is required for OnPrem " +
        "target cluster configuration.")
    }
    gpu.validate()
    if (cpuCores > 0 && gpu.count > cpuCores) {
      throw new IllegalArgumentException(
        s"Target worker has $cpuCores CPU core(s) for ${gpu.count} GPU(s); " +
          "at least one CPU core per GPU is required.")
    }
  }

  /**
   * Helper method to get the NodeInstanceMapKey for this worker. This
   * will be used as the key in the following map:
   * [[com.nvidia.spark.rapids.tool.Platform.getInstanceMapByName]]
   * @return
   */
  def getNodeInstanceMapKey: Option[NodeInstanceMapKey] = {
    if (this.getInstanceType == null || this.getInstanceType.isEmpty) {
      return None
    }
    val gpuCount = this.getGpu.getCount
    if (gpuCount > 0) {
      Some(NodeInstanceMapKey(instanceType = this.getInstanceType, gpuCount = Some(gpuCount)))
    } else {
      Some(NodeInstanceMapKey(instanceType = this.getInstanceType))
    }
  }
}

/**
 * Class to hold the driver instance information for the target cluster.
 */
class DriverInfo (
  @BeanProperty var instanceType: String) {
  def this() = this("")

  def isEmpty: Boolean = {
    instanceType == null || instanceType.isEmpty
  }
}

/**
 * Class to hold the Spark properties specified for the target cluster.
 * This includes enforced properties, preserve/exclude lists, and custom tuning definitions.
 * The tuningDefinitions allow users to define custom Spark properties that will be merged
 * with the default tuning table. This is useful for:
 * - Legacy properties that have been replaced in newer Spark versions
 * - Custom properties specific to a particular Spark distribution
 * - Properties that need special handling or validation
 * Example YAML format:
 * {{{
 * sparkProperties:
 *   enforced:
 *     spark.executor.cores: "8"
 *     spark.executor.memory: "16g"
 *   preserve:
 *     - spark.sql.shuffle.partitions
 *     - spark.sql.files.maxPartitionBytes
 *   exclude:
 *     - spark.rapids.sql.incompatibleDateFormats.enabled
 *     - spark.rapids.sql.explain
 *   tuningDefinitions:
 *     - label: spark.sql.adaptive.shuffle.maxNumPostShufflePartitions
 *       description: alias for spark.sql.adaptive.coalescePartitions.initialPartitionNum
 *       enabled: true
 *       level: job
 *       category: tuning
 *       confType:
 *         name: int
 * }}}
 *
 * Where:
 * - enforced: User-specified values that override any recommendations
 * - preserve: Properties whose values should be preserved from the source application
 * - exclude: Properties that should be excluded from tuning recommendations
 *
 * For more examples, see:
 * - Default tuning definitions: core/src/main/resources/bootstrap/tuningTable.yaml
 * - Target cluster examples: core/docs/sampleFiles/targetClusterInfo/
 */
class SparkProperties(
  @BeanProperty var enforced: util.LinkedHashMap[String, String],
  @BeanProperty var preserve: java.util.List[String],
  @BeanProperty var exclude: java.util.List[String],
  @BeanProperty var tuningDefinitions: java.util.List[TuningEntryDefinition])
  extends ValidatableProperties {
  def this() = this(new util.LinkedHashMap[String, String](),
    new java.util.ArrayList[String](),
    new java.util.ArrayList[String](),
    new java.util.ArrayList[TuningEntryDefinition]())

  def isEmpty: Boolean = {
    enforced.isEmpty && preserve.isEmpty && exclude.isEmpty && tuningDefinitions.isEmpty
  }

  lazy val enforcedPropertiesMap: Map[String, String] = {
    if (enforced.isEmpty) {
      Map.empty
    } else {
      enforced.asScala.toMap
    }
  }

  lazy val preservePropertiesSet: Set[String] = {
    if (preserve.isEmpty) {
      Set.empty
    } else {
      preserve.asScala.toSet
    }
  }

  lazy val excludePropertiesSet: Set[String] = {
    if (exclude.isEmpty) {
      Set.empty
    } else {
      exclude.asScala.toSet
    }
  }

  lazy val tuningDefinitionsMap: Map[String, TuningEntryDefinition] = {
    if (tuningDefinitions.isEmpty) {
      Map.empty
    } else {
      tuningDefinitions.asScala.collect {
        case e if e.isEnabled() => (e.label, e)
      }.toMap
    }
  }

  override def validate(): Unit = {
    if (!isEmpty) {
      // Validate that preserve, exclude and enforced properties do not have overlapping keys.
      val overlappingKeys =
        (preservePropertiesSet & excludePropertiesSet) ++
          (preservePropertiesSet & enforcedPropertiesMap.keySet) ++
          (excludePropertiesSet & enforcedPropertiesMap.keySet)
      if (overlappingKeys.nonEmpty) {
        throw new IllegalArgumentException(
          "Found overlapping keys in preserve, exclude and enforced properties: " +
            s"$overlappingKeys. Please remove the overlapping keys.")
      }
    }
  }
}

/**
 * Class to hold the properties for the target cluster.
 * This class will instantiate from the `--target-cluster-info` YAML file
 * using SnakeYAML.
 *
 * The YAML file can include:
 * - driverInfo: Driver instance configuration
 * - workerInfo: Worker node configuration (instance type or OnPrem resources)
 * - sparkProperties: Spark configuration including enforced properties and tuning definitions
 *
 * Example YAML format:
 * {{{
 * driverInfo:
 *   instanceType: n1-standard-8
 * workerInfo:
 *   instanceType: g2-standard-8
 * sparkProperties:
 *   enforced:
 *     spark.executor.cores: "8"
 *     spark.executor.memory: "16g"
 *   preserve:
 *     - spark.sql.shuffle.partitions
 *     - spark.sql.files.maxPartitionBytes
 *   exclude:
 *     - spark.rapids.sql.incompatibleDateFormats.enabled
 *     - spark.rapids.sql.explain
 *   tuningDefinitions:
 *     - label: spark.sql.adaptive.shuffle.maxNumPostShufflePartitions
 *       description: Legacy property for initial shuffle partitions
 *       enabled: true
 *       level: job
 *       category: tuning
 *       confType:
 *         name: int
 * }}}
 *
 * @see [[org.apache.spark.sql.rapids.tool.util.PropertiesLoader]]
 */
class TargetClusterProps (
  @BeanProperty var driverInfo: DriverInfo,
  @BeanProperty var workerInfo: WorkerInfo,
  @BeanProperty var sparkProperties: SparkProperties) extends ValidatableProperties {

  def this() = this(new DriverInfo(), new WorkerInfo(), new SparkProperties())

  override def validate(): Unit = {
    workerInfo.validate()
    sparkProperties.validate()
    if (workerInfo.isOnpremInfo && !driverInfo.isEmpty) {
      throw new IllegalArgumentException(
        "OnPrem target cluster info does not support specifying driver instance type. " +
          "Please remove the driver instance type from the target cluster info.")
    }
  }

  override def toString: String = {
    s"${getClass.getSimpleName}(workerInfo=$getWorkerInfo, sparkProperties=$getSparkProperties)"
  }
}
