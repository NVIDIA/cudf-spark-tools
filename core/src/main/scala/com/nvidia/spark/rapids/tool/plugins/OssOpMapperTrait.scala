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

package com.nvidia.spark.rapids.tool.plugins

import scala.util.matching.Regex

import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

import org.apache.spark.sql.execution
import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.util.UTF8Source
import org.apache.spark.sql.rapids.tool.util.stubs.{EnclosingClusterPolicy, SparkPlanInfo}
import org.apache.spark.sql.rapids.tool.util.stubs.SparkPlanExtensions.UpStreamSparkPlanInfoOps
import org.apache.spark.sql.rapids.tool.util.stubs.rapids.RAPIDSSparkPlanInfo


/**
 * Trait defining the contract for mapping platform-specific execution operators to OSS Spark
 * equivalents.
 *
 * This trait provides the abstraction layer needed to convert execution plans from different
 * platforms (e.g., RAPIDS GPU, Databricks Photon, Auron Native Spark) into OSS Spark-compatible
 * representations. Each platform-specific mapper implements this trait to handle its unique
 * operator naming conventions and execution semantics.
 *
 * The mapping process involves:
 * 1. Detection: Identifying whether a plan node belongs to this platform
 * 2. Transformation: Converting platform-specific names/descriptions to OSS equivalents
 * 3. Construction: Building a platform-aware SparkPlanInfo that preserves both representations
 *
 * Implementations are registered in SparkPlanExtensions.converters and are tried in order
 * during plan conversion.
 */
trait OssOpMapperTrait {
  /**
   * Map of platform-specific operator names to their OSS Spark equivalents.
   *
   * For example, GPU operations might map "GpuProject" → "Project", or Photon might map
   * "PhotonProject" → "Project". This mapping is used to translate operator names during
   * plan conversion.
   */
  val ossSparkMapping: Map[String, String]

  /**
   * Condition that determines if this mapper should be active for a given application context.
   *
   * This allows mappers to be conditionally enabled based on application properties. For example,
   * the GPU mapper checks if GPU mode is enabled, while the Photon mapper checks if the
   * application is running on Databricks.
   */
  val appCtxtCondition: ConditionTrait[AppBase]

  /**
   * Builds the operator mapping from platform-specific names to OSS Spark equivalents.
   *
   * Implementations may load this mapping from JSON files, compute it dynamically, or return
   * an empty map if simple pattern-based transformation is sufficient.
   *
   * @return Map from platform operator names to OSS Spark operator names
   */
  def buildOssSparkMapping(): Map[String, String]

  /**
   * Determines if this mapper accepts the given plan node, considering the application context.
   *
   * This is the primary entry point for plan node detection. It first checks if the application
   * context matches (via appCtxtCondition), then delegates to acceptPlanInfo to check the
   * specific plan node.
   *
   * @param planInfo The upstream Spark plan node to evaluate
   * @param app Optional application context for conditional acceptance
   * @return true if this mapper should handle the plan node, false otherwise
   */
  def acceptsPlanInfo(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo,
      app: Option[AppBase]): Boolean = {
    app match {
      case Some(a) if appCtxtCondition.eval(a) =>
        acceptPlanInfo(planInfo)
      case _ =>
        false
    }
  }

  /**
   * Determines if this mapper accepts the given plan node based solely on the node itself.
   *
   * This method performs platform-specific detection logic, such as checking if the node name
   * matches a particular pattern (e.g., starts with "Gpu", "Photon", or "Native").
   *
   * @param planInfo The upstream Spark plan node to evaluate
   * @return true if this mapper should handle the plan node, false otherwise
   */
  def acceptPlanInfo(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo): Boolean

  /**
   * Maps platform-specific plan data (node name and description) to OSS Spark equivalents.
   *
   * This method transforms both the operator name and its description string to their OSS
   * counterparts. For example, "GpuProject" might become "Project", and descriptions containing
   * GPU-specific details are converted to standard Spark descriptions.
   *
   * @param planInfo The upstream Spark plan node to map
   * @return Tuple of (OSS node name, OSS description)
   */
  def mapPlanDataToOss(planInfo: org.apache.spark.sql.execution.SparkPlanInfo): (String, String) = {
    (mapContentToOss(planInfo.nodeName), mapContentToOss(planInfo.simpleString))
  }

  /**
   * Determines how the converted node interacts with an active enclosing execution cluster.
   * Platform mappers should override this only when the canonical Spark name cannot distinguish
   * whether the materialized operator belongs to its parent execution cluster.
   * A mapper returning a policy other than Inherit must construct a PWSparkPlanInfo subtype so
   * that the policy survives conversion.
   *
   * @param planInfo The upstream Spark plan node to classify
   * @return The policy to carry into shared graph construction
   */
  def enclosingClusterPolicy(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo): EnclosingClusterPolicy = {
    EnclosingClusterPolicy.Inherit
  }

  /**
   * Converts an upstream Spark plan node into a platform-aware SparkPlanInfo.
   *
   * This is the core transformation method that creates a SparkPlanInfo object containing both
   * the original platform-specific information and the mapped OSS Spark equivalents. The result
   * enables tools to analyze plans from different platforms using a unified interface.
   *
   * @param planInfo The upstream Spark plan node to convert
   * @param app The application context providing platform-specific information
   * @return A platform-aware SparkPlanInfo with both original and OSS representations
   */
  def toPlatformAwarePlan(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo,
      app: AppBase): SparkPlanInfo

  /**
   * Maps platform-specific content (operator names, descriptions) to OSS Spark equivalents.
   *
   * This method performs string-based transformations to convert platform-specific terminology
   * to standard Spark terminology. Implementations use pattern matching, string replacement, or
   * lookup tables to perform the conversion.
   *
   * Examples:
   * - GPU: "GpuProject" → "Project"
   * - Photon: "PhotonProject" → "Project"
   * - Auron: "NativeProject" → "Project"
   *
   * @param inputStr The platform-specific string to convert
   * @return The OSS Spark equivalent string
   */
  def mapContentToOss(inputStr: String): String
}

/**
 * Extension of OssOpMapperTrait that loads operator mappings from JSON files.
 *
 * This trait provides a file-based approach to defining operator mappings, which is useful
 * for platforms with complex or frequently changing operator sets. The JSON file should
 * contain a map where keys are platform-specific operator names and values are lists of
 * possible OSS Spark equivalents.
 *
 * JSON format example:
 * {{{
 * {
 *   "PhotonProject": ["Project"],
 *   "PhotonFilter": ["Filter"],
 *   "PhotonBroadcastHashJoin": ["BroadcastHashJoin"]
 * }
 * }}}
 *
 * Note: Currently only the first mapping in each list is used, as the tools cannot yet
 * differentiate between multiple possible mappings for the same operator.
 */
trait OssOpMapperFromFileTrait extends OssOpMapperTrait {
  /**
   * Path to the JSON file containing operator mappings.
   *
   * The path should be relative to the resources directory. For example:
   * "parser/photon/photon-oss-op-map.json"
   */
  def mappingFilePath: String

  /**
   * Lazily loaded operator mapping from the JSON file.
   *
   * The mapping is loaded once on first access and cached for subsequent use.
   */
  override lazy val ossSparkMapping: Map[String, String] = buildOssSparkMapping()

  /**
   * Builds the operator mapping by loading and parsing the JSON file.
   *
   * This method:
   * 1. Reads the JSON file from resources
   * 2. Parses it into a Map from String to List of String
   * 3. Takes the first element from each list (head)
   * 4. Returns a simplified Map from String to String
   *
   * @return Map from platform operator names to OSS Spark operator names
   */
  def buildOssSparkMapping(): Map[String, String] = {
    val jsonString = UTF8Source.fromResource(mappingFilePath).mkString
    val json = JsonMethods.parse(jsonString)
    // Implicitly define JSON formats for deserialization using DefaultFormats
    implicit val formats: Formats = DefaultFormats
    // Extract and deserialize the JValue object into a Map[String, String]
    // TODO: Currently, only the first mapping in the list is used.
    //       This limitation exists because we cannot differentiate between
    //       these operators in the SparkPlan.
    json.extract[Map[String, List[String]]]
      .iterator
      .map { case (k, v) => k -> v.head }
      .toMap
  }
}

/**
 * Default fallback mapper for standard OSS Spark execution plans.
 *
 * This object serves as the identity mapper that handles nodes which don't belong to any
 * specific platform (GPU, Photon, Auron, etc.). It accepts all plan nodes and returns them
 * unchanged, as they are already in OSS Spark format.
 *
 * This mapper is used as the final fallback in the conversion chain. When no other
 * platform-specific mapper accepts a node, OssOpMapper is applied to ensure every node
 * can be processed.
 *
 * Characteristics:
 * - Always active (AlwaysTrueCondition)
 * - Accepts all plan nodes
 * - No mapping required (identity transformation)
 * - Returns nodes unchanged in their original form
 */
object OssOpMapper extends OssOpMapperTrait {
  /**
   * Empty mapping since no transformation is needed for OSS Spark nodes.
   * Standard Spark operators are already in their canonical form.
   */
  override lazy val ossSparkMapping: Map[String, String] = buildOssSparkMapping()

  /**
   * @inheritdoc
   * Returns an empty map as no operator mapping is required.
   */
  override def buildOssSparkMapping(): Map[String, String] = Map.empty

  /**
   * Always active condition - accepts nodes from any application context.
   * This ensures OssOpMapper can serve as a universal fallback.
   */
  override val appCtxtCondition: ConditionTrait[AppBase] =
    new AlwaysTrueCondition[AppBase]()

  /**
   * @inheritdoc
   * Accepts all plan nodes from any application context.
   */
  override def acceptsPlanInfo(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo,
      app: Option[AppBase]): Boolean = true

  /**
   * @inheritdoc
   * Creates a standard SparkPlanInfo with no platform-specific transformations.
   * The node name and description remain unchanged as they are already in OSS format.
   */
  override def toPlatformAwarePlan(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo,
      app: AppBase): SparkPlanInfo = {
    new SparkPlanInfo(
      planInfo.nodeName,
      planInfo.simpleString,
      children = planInfo.children.map(_.asPlatformAware(app)),
      metadata = planInfo.metadata,
      metrics = planInfo.metrics)
  }

  /**
   * @inheritdoc
   * Identity transformation - returns the input string unchanged.
   */
  override def mapContentToOss(inputStr: String): String = inputStr

  /**
   * @inheritdoc
   * Accepts all plan nodes as they are already in OSS Spark format.
   */
  override def acceptPlanInfo(planInfo: execution.SparkPlanInfo): Boolean = true
}

/**
 * Mapper for RAPIDS GPU-accelerated execution operators.
 *
 * This object handles conversion of RAPIDS GPU execution plans to OSS Spark equivalents.
 * GPU operators follow the naming convention "Gpu<OperatorName>" (e.g., GpuProject, GpuFilter)
 * and are mapped to their CPU counterparts by stripping the "Gpu" prefix.
 *
 * Key characteristics:
 * - Detects GPU nodes using pattern matching (Gpu[A-Z][a-zA-Z]+)
 * - Only active when GPU mode is enabled in the application
 * - Creates RAPIDSSparkPlanInfo preserving both GPU and OSS representations
 * - Simple prefix-stripping transformation (GpuProject → Project)
 *
 * Example transformations:
 * - GpuProject → Project
 * - GpuFilter → Filter
 * - GpuColumnarToRow → ColumnarToRow
 * - GpuBroadcastHashJoin → BroadcastHashJoin
 */
object GpuOssOpMapper extends OssOpMapperTrait {
  /**
   * Regex pattern to identify GPU operators.
   * Matches operator names starting with "Gpu" followed by an uppercase letter and
   * additional alphanumeric characters (e.g., GpuProject, GpuFilter).
   */
  private val GPU_PATTERN: Regex = "Gpu[A-Z][a-zA-Z]+".r

  /**
   * Empty mapping - GPU operators use pattern-based transformation instead of lookup tables.
   * The "Gpu" prefix is simply stripped to get the OSS equivalent.
   */
  override lazy val ossSparkMapping: Map[String, String] = buildOssSparkMapping()

  /**
   * Checks if a node name represents a GPU operator by matching against GPU_PATTERN.
   *
   * @param nodeName The operator name to check
   * @return true if the name matches the GPU pattern, false otherwise
   */
  def isGpuNode(nodeName: String): Boolean = GPU_PATTERN.findFirstIn(nodeName).isDefined

  /**
   * @inheritdoc
   * Creates a RAPIDSSparkPlanInfo containing both the original GPU operator information
   * and the mapped OSS Spark equivalent. This allows tools to analyze GPU plans using
   * standard Spark operator semantics while preserving GPU-specific details.
   */
  override def toPlatformAwarePlan(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo,
      app: AppBase): SparkPlanInfo = {
    val (ossPlanName, ossPlanDesc) = mapPlanDataToOss(planInfo)
    RAPIDSSparkPlanInfo(
      actualName = planInfo.nodeName,
      actualDesc = planInfo.simpleString,
      sparkName = ossPlanName,
      sparkDesc = ossPlanDesc,
      children = planInfo.children.map(_.asPlatformAware(app)),
      metadata = planInfo.metadata,
      metrics = planInfo.metrics,
      enclosingClusterPolicy = enclosingClusterPolicy(planInfo))
  }

  /**
   * Condition that checks if GPU mode is enabled in the application.
   * For profiling applications (ApplicationInfo), checks the gpuMode flag.
   * For other application types, defaults to false.
   */
  override val appCtxtCondition: ConditionTrait[AppBase] = {
    case appInfo: ApplicationInfo => appInfo.gpuMode
    case _ => false
  }

  /**
   * @inheritdoc
   * Accepts plan nodes that match the GPU operator pattern (Gpu[A-Z][a-zA-Z]+).
   */
  override def acceptPlanInfo(planInfo: execution.SparkPlanInfo): Boolean = {
    isGpuNode(planInfo.nodeName)
  }

  /**
   * @inheritdoc
   * Strips the "Gpu" prefix from all GPU operator names in the input string.
   *
   * Uses GPU_PATTERN to find all occurrences of GPU operators and removes the "Gpu"
   * prefix to get the OSS Spark equivalent. For example:
   * - "GpuProject [columns]" → "Project [columns]"
   * - "GpuFilter (condition)" → "Filter (condition)"
   *
   * @param inputStr String containing GPU operator names
   * @return String with GPU prefixes removed
   */
  override def mapContentToOss(inputStr: String): String = {
    GPU_PATTERN.replaceAllIn(inputStr, m => m.matched.stripPrefix("Gpu"))
  }

  /**
   * @inheritdoc
   * Returns an empty map as GPU operators don't require a lookup table.
   * The transformation is performed by simple prefix stripping via pattern matching.
   */
  override def buildOssSparkMapping(): Map[String, String] = Map.empty
}
