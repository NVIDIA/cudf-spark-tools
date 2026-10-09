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

package org.apache.spark.sql.rapids.tool.util.stubs

import org.apache.spark.sql.execution.metric.SQLMetricInfo
import org.apache.spark.sql.rapids.tool.annotation.ToolsReflection


/**
 * Carries a platform mapper's decision about whether a node inherits membership in an execution
 * cluster that is already open while the shared plan graph is being built.
 *
 * Platform nodes are first mapped to canonical Spark names. If graph traversal is currently
 * inside a `WholeStageCodegen`-like cluster, [[EnclosingClusterPolicy.Inherit]] keeps that parent
 * cluster active while the canonical-name rules handle the node. [[EnclosingClusterPolicy.Break]]
 * clears only the incoming cluster context before applying those same rules. The node is therefore
 * not recorded as a child of the parent cluster, but it or its descendants may still open their
 * own clusters through normal canonical-name dispatch.
 *
 * For example, a `PhotonShuffleMapStage` maps to `WholeStageCodegen` and opens a cluster. Its
 * `PhotonShuffleExchangeSink` is fused work within that stage and inherits the cluster. A
 * `PhotonShuffleExchangeSource` reads another stage's output, so it breaks the enclosing cluster
 * and is not scored as fused work in the shuffle-map stage.
 *
 * This policy does not decide whether a node is a cluster. The canonical mapped name remains the
 * authority for that decision.
 */
sealed trait EnclosingClusterPolicy


object EnclosingClusterPolicy {
  /** Keep any active parent cluster while canonical-name dispatch handles this node. */
  case object Inherit extends EnclosingClusterPolicy

  /** Clear any active parent cluster before canonical-name dispatch handles this node. */
  case object Break extends EnclosingClusterPolicy
}

/**
 * Represents execution plan information with platform awareness capabilities.
 *
 * This code is copied from org.apache.spark.sql.execution.SparkPlanInfo
 * with changes to handle information about non-Spark plans (e.g., Photon, RAPIDS GPU).
 * Without this special handling, the default SparkPlanInfo would not be efficient in
 * representing platform-specific execution plans that may have different node names
 * and descriptions than their Spark equivalents.
 *
 * The class implements PlatformAwarePlanTrait to expose platform-specific information
 * while maintaining compatibility with standard Spark execution plans.
 *
 * @param nodeName The name of the execution plan node
 * @param simpleString A simple string representation of the plan
 * @param children Child execution plan nodes in the plan tree
 * @param metadata Additional metadata associated with the plan node
 * @param metrics SQL metrics collected for this plan node
 */
@ToolsReflection("Spark OSS 3.5.x", "Defines the Spark PlanInfo with platform awareness")
class SparkPlanInfo(
    val nodeName: String,
    val simpleString: String,
    val children: Seq[SparkPlanInfo],
    val metadata: Map[String, String],
    val metrics: Seq[SQLMetricInfo]) extends PlatformAwarePlanTrait {

  /** Native Spark nodes keep the graph builder's current cluster context. */
  def enclosingClusterPolicy: EnclosingClusterPolicy = EnclosingClusterPolicy.Inherit

  /**
   * Computes hash code based on the simple string representation.
   * The simpleString hashCode is sufficient to distinguish different plans within a plan tree.
   */
  override def hashCode(): Int = {
    // hashCode of simpleString should be good enough to distinguish the plans from each other
    // within a plan
    platformDesc.hashCode
  }

  /**
   * Compares this plan with another for equality.
   * Two plans are equal if they have the same platform name, platform description, and children.
   *
   * @param other The object to compare with
   * @return true if the plans are equal, false otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case o: SparkPlanInfo =>
      platformName == o.platformName && platformDesc == o.platformDesc && children == o.children
    case _ => false
  }

  /**
   * Returns the platform name for this plan.
   * For standard Spark plans, this is the node name.
   *
   * @return The platform identifier
   */
  override def platformName: String = nodeName

  /**
   * Returns the platform description for this plan.
   * For standard Spark plans, this is the simple string representation.
   *
   * @return The platform description
   */
  override def platformDesc: String = simpleString

  /**
   * Recursively checks if this plan or any of its descendants is of the specified platform type.
   *
   * @param targetClass The target platform class to search for (e.g., classOf[PhotonSparkPlanInfo])
   * @return true if this plan or any child plan is an instance of the target class
   */
  def hasPlatformOps(targetClass: Class[_ <: SparkPlanInfo]): Boolean = {
    getClass.equals(targetClass) ||
      children.exists(_.hasPlatformOps(targetClass))
  }
}

/**
 * Platform-Wrapper SparkPlanInfo that maintains both platform-specific and Spark-equivalent
 * representations of an execution plan node.
 *
 * This class enables dual representation where platform-specific execution engines
 * (e.g., Photon, RAPIDS GPU) can preserve their native node names and descriptions
 * while also providing equivalent Spark node information for compatibility and analysis.
 *
 * This is the base class for platform-specific plan implementations like PhotonSparkPlanInfo
 * and RAPIDSSparkPlanInfo.
 *
 * @param actualName The actual platform-specific node name (e.g., "PhotonProject", "GpuProject")
 * @param actualDesc The actual platform-specific description
 * @param sparkName The equivalent Spark node name (e.g., "Project")
 * @param sparkDesc The equivalent Spark description
 * @param children Child execution plan nodes
 * @param metadata Additional metadata associated with the plan node
 * @param metrics SQL metrics collected for this plan node
 * @param enclosingClusterPolicy How this node interacts with an active enclosing cluster
 */
class PWSparkPlanInfo(
    actualName: String,
    actualDesc: String,
    sparkName: String,
    sparkDesc: String,
    children: Seq[SparkPlanInfo],
    metadata: Map[String, String],
    metrics: Seq[SQLMetricInfo],
    override val enclosingClusterPolicy: EnclosingClusterPolicy
) extends SparkPlanInfo(sparkName, sparkDesc, children, metadata, metrics)
  with PlatformAwarePlanTrait {
  /**
   * Returns the actual platform-specific node name.
   *
   * @return The native platform node name
   */
  override def platformName: String = actualName

  /**
   * Returns the actual platform-specific description.
   *
   * @return The native platform description
   */
  override def platformDesc: String = actualDesc
}
