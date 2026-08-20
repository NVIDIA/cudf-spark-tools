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

import com.nvidia.spark.rapids.tool.planparser.ReadParser
import com.nvidia.spark.rapids.tool.planparser.auron.AuronOssOpMapper
import com.nvidia.spark.rapids.tool.planparser.db.PhotonOssOpMapper
import com.nvidia.spark.rapids.tool.planparser.hive.HiveParseHelper
import com.nvidia.spark.rapids.tool.plugins.{GpuOssOpMapper, OssOpMapper, OssOpMapperTrait}

import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.util.stubs.db.PhotonSparkPlanInfo
import org.apache.spark.sql.rapids.tool.util.stubs.rapids.RAPIDSSparkPlanInfo

/**
 * Provides extension methods for converting upstream Spark execution plans to platform-aware plans.
 *
 * This object defines implicit conversions that enable the cuDF plugin tools to process execution plans
 * from different platforms (e.g., Databricks Photon, standard Spark) and convert them into
 * a unified representation that maintains both platform-specific and Spark-equivalent information.
 *
 * The primary use case is to take execution plans from the upstream Spark SQL engine
 * (org.apache.spark.sql.execution.SparkPlanInfo) and transform them into platform-aware
 * representations that can be analyzed and compared across different execution engines.
 */
object SparkPlanExtensions {

  /**
   * Ordered sequence of platform-specific operator mappers used for plan conversion.
   *
   * This sequence defines the registered converters that transform platform-specific execution
   * nodes into OSS Spark equivalents. Each converter implements OssOpMapperTrait and is
   * responsible for:
   * - Detecting whether a plan node belongs to its platform (via acceptsPlanInfo)
   * - Mapping platform-specific operator names to OSS Spark operator names
   * - Converting platform-specific descriptions to OSS equivalents
   *
   * The converters are tried in order during plan conversion. The first converter that accepts
   * a plan node is used to perform the transformation. Currently registered converters:
   * - GpuOssOpMapper: Handles RAPIDS GPU-accelerated operators (e.g., GpuProject → Project)
   * - AuronOssOpMapper: Handles Auron/Native Spark operators (e.g., NativeProject → Project)
   * - PhotonOssOpMapper: Handles Databricks Photon operators (e.g., PhotonProject → Project)
   *
   * If no converter accepts a plan node, it falls back to OssOpMapper which returns the node
   * unchanged as a standard Spark node.
   *
   * New platform converters can be added by implementing OssOpMapperTrait and adding them
   * to this sequence. The order matters when multiple converters might match the same node.
   */
  val converters: Seq[OssOpMapperTrait] =
    Seq(
      GpuOssOpMapper,   // RAPIDS converter. Only used for RAPIDS plans.
      AuronOssOpMapper, // Auron/Native Spark converter.
      PhotonOssOpMapper // Databricks Photon converter.
    )

  /**
   * Implicit class that adds extension methods to upstream Spark's SparkPlanInfo.
   *
   * This enrichment enables conversion of standard Spark execution plans into platform-aware
   * representations through the `asPlatformAware` method.
   *
   * @param planInfo The upstream Spark execution plan to be converted
   */
  implicit class UpStreamSparkPlanInfoOps(
      val planInfo: org.apache.spark.sql.execution.SparkPlanInfo) {

    /**
     * Converts an upstream Spark execution plan into a platform-aware representation.
     *
     * This method analyzes the plan node and determines if it originates from a
     * platform-specific execution engine (e.g., Databricks Photon, Apache Auron).
     * If so, it creates a dual-representation plan that preserves both the
     * platform-specific node information and the equivalent Spark node information.
     * For standard Spark nodes, it creates a standard SparkPlanInfo representation.
     *
     * The conversion is recursive - all child nodes in the plan tree are also converted.
     *
     * @param app The application context providing metadata and configuration
     * @return A platform-aware SparkPlanInfo that may be:
     *         - PhotonSparkPlanInfo if the node is from Databricks Photon engine
     *         - AuronSparkPlanInfo if the node is from Apache Auron engine
     *         - Standard SparkPlanInfo for native Spark nodes
     */
    def asPlatformAware(app: AppBase): SparkPlanInfo = {
      converters.find(_.acceptsPlanInfo(planInfo, Option(app)))
        .getOrElse(OssOpMapper)
        .toPlatformAwarePlan(planInfo, app)
    }
  }

  implicit class SparkPlanInfoOps(
      val pInfo: SparkPlanInfo) {

    def isPhoton: Boolean = {
      pInfo match {
        case _: PhotonSparkPlanInfo => true
        case _ => false
      }
    }

    def isRAPIDS: Boolean = {
      pInfo match {
        case _: RAPIDSSparkPlanInfo => true
        case _ => false
      }
    }


    /**
     * Recursively collects all SparkPlanInfo nodes that match the given predicate.
     *
     * This method traverses the plan tree and collects all nodes (including the current node
     * and all descendants) that satisfy the provided predicate function.
     *
     * @param predicate A function that determines whether a SparkPlanInfo node should be included
     * @param skipReusedSubquery If true, filters out "ReusedSubquery" nodes to avoid duplicate
     *                           results
     * @return A sequence of SparkPlanInfo nodes that match the predicate
     */
    def collectPlans(predicate: SparkPlanInfo => Boolean,
                     skipReusedSubquery: Boolean = true): Seq[SparkPlanInfo] = {
      // Filter out "ReusedSubquery" nodes as they just point to other nodes.
      // Otherwise, the planInfo will show up twice in the recursive results.
      val childrenToProcess = if (skipReusedSubquery) {
        pInfo.children.filterNot(_.nodeName.startsWith("ReusedSubquery"))
      } else {
        pInfo.children
      }

      val childRes = childrenToProcess.flatMap(_.collectPlans(predicate, skipReusedSubquery))

      if (predicate(pInfo)) {
        childRes :+ pInfo
      } else {
        childRes
      }
    }

    /**
     * Recursive call to get all the SparkPlanInfo that have a schema attached to it.
     * This is mainly used for V1 ReadSchema.
     * @return A list of SparkPlanInfo that have a schema attached to it.
     */
    def getPlansWithSchema: Seq[SparkPlanInfo] = {
      collectPlans(plan =>
        plan.metadata != null && plan.metadata.contains(ReadParser.METAFIELD_TAG_READ_SCHEMA))
    }

    /**
     * Recursively finds all Hive table scan nodes in the execution plan.
     *
     * Hive table scans are identified by their node name starting with "scan hive"
     * (case-insensitive). This method traverses the plan tree and collects all
     * nodes that match this pattern.
     *
     * @return A sequence of SparkPlanInfo nodes representing Hive table scans
     */
    def getPlansWithHiveScan: Seq[SparkPlanInfo] = {
      collectPlans(plan => HiveParseHelper.isHiveTableScanNode(plan.nodeName))
    }
  }
}
