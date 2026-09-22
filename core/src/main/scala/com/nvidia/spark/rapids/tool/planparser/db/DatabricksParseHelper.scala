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

package com.nvidia.spark.rapids.tool.planparser.db

import java.nio.file.Paths
import java.util.concurrent.TimeUnit

import scala.util.control.NonFatal
import scala.util.matching.Regex

import com.nvidia.spark.rapids.tool.plugins.{AppPropVersionExtractorTrait, ConditionTrait, OssOpMapperFromFileTrait, PropConditionOnSparkExtTrait}
import com.nvidia.spark.rapids.tool.profiling.SQLAccumProfileResults
import com.nvidia.spark.rapids.tool.views.IoMetrics
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.parse

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution
import org.apache.spark.sql.rapids.tool.{AppBase, UnsupportedMetricNameException}
import org.apache.spark.sql.rapids.tool.util.stubs.{EnclosingClusterPolicy, SparkPlanInfo}
import org.apache.spark.sql.rapids.tool.util.stubs.SparkPlanExtensions.UpStreamSparkPlanInfoOps
import org.apache.spark.sql.rapids.tool.util.stubs.db.PhotonSparkPlanInfo


/**
 * Mapper for Databricks Photon execution operators.
 *
 * Photon is Databricks' vectorized execution engine that uses operators with "Photon" prefix
 * (e.g., PhotonProject, PhotonFilter). This mapper transforms Photon operators to OSS Spark
 * equivalents for analysis while preserving the original Photon execution information.
 *
 * Key differences from parent:
 * - Loads operator mappings from JSON file (databricks-13_3.json)
 * - Pattern matches "Photon*" operators (any alphabetic characters after "Photon")
 * - Only active when running on Databricks with Photon enabled (app.isPhoton)
 * - Creates PhotonSparkPlanInfo to maintain Photon-specific context
 *
 * Example transformations:
 * - PhotonProject → Project
 * - PhotonHashAggregate → HashAggregate
 * - PhotonBroadcastHashJoin → BroadcastHashJoin
 *
 * Note: Currently uses a single mapping file for all Databricks/Photon versions.
 * TODO: Create separate mapping files for different versions.
 */
object PhotonOssOpMapper extends OssOpMapperFromFileTrait {
  /** Pattern matching Photon operators: "Photon" followed by any alphabetic characters */
  private val PHOTON_PATTERN: Regex = "Photon[A-Z][a-zA-Z]+".r

  /** Photon operators that must be constructed outside an active enclosing cluster. */
  private[tool] val CLUSTER_BOUNDARY_OPERATORS = Set("PhotonShuffleExchangeSource")

  /** Directory containing Photon operator mapping files */
  private val OPS_MAPPING_DIR = "parser/photon"

  // TODO: Create separate mapping file for different Photon/Databricks versions
  /** Default operator mapping file for Databricks 13.3 */
  private val DEFAULT_OPS_MAPPING_FILE = "databricks-13_3.json"

  override def mappingFilePath: String =
    Paths.get(OPS_MAPPING_DIR, DEFAULT_OPS_MAPPING_FILE).toString

  private def findPhotonOperator(inputStr: String): Option[String] = {
    PHOTON_PATTERN.findFirstIn(inputStr)
  }

  /** Checks if a node name matches the Photon operator pattern */
  def isPhotonNode(nodeName: String): Boolean = findPhotonOperator(nodeName).isDefined

  override def enclosingClusterPolicy(
      planInfo: execution.SparkPlanInfo): EnclosingClusterPolicy = {
    findPhotonOperator(planInfo.nodeName) match {
      case Some(operator) if CLUSTER_BOUNDARY_OPERATORS.contains(operator) =>
        EnclosingClusterPolicy.Break
      case _ =>
        super.enclosingClusterPolicy(planInfo)
    }
  }

  /**
   * Creates a PhotonSparkPlanInfo preserving both Photon and OSS representations.
   * Uses PhotonSparkPlanInfo instead of generic SparkPlanInfo to maintain Photon context.
   */
  override def toPlatformAwarePlan(
    planInfo: org.apache.spark.sql.execution.SparkPlanInfo, app: AppBase): SparkPlanInfo = {
    val (ossPlanName, ossPlanDesc) = mapPlanDataToOss(planInfo)
    // Create a PhotonSparkPlanInfo that maintains both Photon and Spark representations
    PhotonSparkPlanInfo(
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
   * Maps Photon operator names to OSS equivalents using the loaded JSON mapping.
   * Performs lookup-based transformation (e.g., PhotonProject → Project).
   *
   * @param inputStr String potentially containing Photon operator names
   * @return String with Photon operators replaced by OSS equivalents
   */
  def mapContentToOss(inputStr: String): String = {
    PHOTON_PATTERN.replaceAllIn(inputStr, m => ossSparkMapping.getOrElse(m.matched, m.matched))
  }

  /** Only active when Photon is enabled in the Databricks application */
  override val appCtxtCondition: ConditionTrait[AppBase] =
    (app: AppBase) => app.dbPlugin.isPhotonEnabled

  override def acceptPlanInfo(planInfo: execution.SparkPlanInfo): Boolean = {
    isPhotonNode(planInfo.nodeName)
  }
}

object DBVersionExtractor extends AppPropVersionExtractorTrait {
  /** Property key containing the Databricks runtime version */
  val DB_SPARK_VERSION_KEY = "spark.databricks.clusterUsageTags.sparkVersion"
  /**
   * Extracts Databricks runtime version directly from property value.
   * The version string is the value of spark.databricks.clusterUsageTags.sparkVersion.
   */
  override def extractVersion(properties: collection.Map[String, String]): Option[String] = {
    properties.get(DB_SPARK_VERSION_KEY)
  }
}

object PhotonParseHelper extends PropConditionOnSparkExtTrait {
  // scalastyle:off
  /**
   * Photon-specific metric labels that serve as alternatives to standard Spark metrics.
   * Photon uses different terminology for some metrics:
   * - "cumulative time" instead of "scan time"
   * - "peak memory usage" instead of "peak execution memory"
   * - "part of shuffle file write" instead of "shuffle write time"
   */
  val PHOTON_METRIC_CUMULATIVE_TIME_LABEL = "cumulative time"               // Alternative for "scan time"
  val PHOTON_METRIC_PEAK_MEMORY_LABEL = "peak memory usage"                 // Alternative for "peak execution memory"
  val PHOTON_METRIC_SHUFFLE_WRITE_TIME_LABEL = "part of shuffle file write" // Alternative for "shuffle write time"
  // scalastyle:on

  /**
   * Spark properties used to identify Photon applications.
   * A map where keys are Spark extension property names and values are
   * regex patterns to match against the property values.
   */
  override val extensionRegxMap: Map[String, String] = Map(
    "spark.databricks.clusterUsageTags.sparkVersion" -> ".*-photon-.*",
    "spark.databricks.clusterUsageTags.effectiveSparkVersion" -> ".*-photon-.*",
    "spark.databricks.clusterUsageTags.sparkImageLabel" -> ".*-photon-.*",
    "spark.databricks.clusterUsageTags.runtimeEngine" -> "PHOTON"
  )

  /**
   * Checks if an accumulator represents a Photon I/O metric.
   * Photon I/O metrics are identified by "cumulative time" name on Scan nodes.
   */
  def isPhotonIoMetric(accum: SQLAccumProfileResults): Boolean =
    accum.name == PHOTON_METRIC_CUMULATIVE_TIME_LABEL && accum.nodeName.contains("Scan")

  /**
   * Updates I/O metrics for Photon applications using Photon-specific metric names.
   * Converts Photon's "cumulative time" (in nanoseconds) to scanTime (in milliseconds).
   */
  @throws[UnsupportedMetricNameException]
  def updatePhotonIoMetric(accum: SQLAccumProfileResults, ioMetrics: IoMetrics): Unit = {
    accum.name match {
      case PHOTON_METRIC_CUMULATIVE_TIME_LABEL if accum.nodeName.contains("Scan") =>
        ioMetrics.scanTime = TimeUnit.NANOSECONDS.toMillis(accum.total)
      case _ => throw UnsupportedMetricNameException(accum.name)
    }
  }
}

/**
 * Utilities for detecting and handling Databricks Photon applications.
 *
 * Provides helper methods for:
 * - Detecting Photon apps via Spark properties
 * - Parsing Databricks cluster metadata (JobId, cluster tags)
 * - Handling Photon-specific metrics that differ from standard Spark metrics
 *
 * Photon Detection:
 * An application is identified as Photon if any of the PHOTON_SPARK_PROPS keys contain
 * Photon-related values (e.g., sparkVersion contains "-photon-", runtimeEngine is "PHOTON").
 */
object DatabricksParseHelper extends Logging {
  /** Property keys for Databricks cluster metadata */
  val PROP_ALL_TAGS_KEY = "spark.databricks.clusterUsageTags.clusterAllTags"
  val PROP_TAG_CLUSTER_ID_KEY = "spark.databricks.clusterUsageTags.clusterId"
  val PROP_TAG_CLUSTER_NAME_KEY = "spark.databricks.clusterUsageTags.clusterName"
  val PROP_WORKER_TYPE_ID_KEY = "spark.databricks.workerNodeTypeId"
  val PROP_DRIVER_TYPE_ID_KEY = "spark.databricks.driverNodeTypeId"

  /** Sub-properties extracted from cluster tags JSON */
  val SUB_PROP_CLUSTER_ID = "ClusterId"
  val SUB_PROP_JOB_ID = "JobId"
  val SUB_PROP_RUN_NAME = "RunName"


  /** Extracts Spark version from cluster properties */
  def getSparkVersion(properties: collection.Map[String, String]): String = {
    DBVersionExtractor.extractVersion(properties).getOrElse("")
  }

  /**
   * Extracts JobId from cluster name string.
   *
   * Databricks cluster names for jobs follow the pattern: "job-{jobId}-run-{runId}"
   * This method parses the cluster name and extracts the JobId portion.
   *
   * @param clusterNameString Cluster name (e.g., "job-215-run-1")
   * @return Optional JobId if found in the expected format
   */
  def parseClusterNameForJobId(clusterNameString: String): Option[String] = {
    var jobId: Option[String] = None
    val splitArr = clusterNameString.split("-")
    val jobIdx = splitArr.indexOf("job")
    if (jobIdx != -1 && splitArr.length > jobIdx + 1) {
      jobId = Some(splitArr(jobIdx + 1))
    }
    jobId
  }

  /**
   * Parses Databricks cluster tags from JSON format to key-value map.
   *
   * Cluster tags contain metadata like Vendor, Creator, ClusterId, JobId, RunName, etc.
   * The input is a JSON array of objects with "key" and "value" properties.
   *
   * @param clusterTag JSON string from spark.databricks.clusterUsageTags.clusterAllTags
   * @return Map of cluster tag keys to values, or empty map if parsing fails
   */
  def parseClusterTags(clusterTag: String): Map[String, String] = {
    // clusterTags will be in this format -
    // [{"key":"Vendor","value":"Databricks"},
    // {"key":"Creator","value":"abc@company.com"},{"key":"ClusterName",
    // "value":"job-215-run-1"},{"key":"ClusterId","value":"0617-131246-dray530"},
    // {"key":"JobId","value":"215"},{"key":"RunName","value":"test73longer"},
    // {"key":"DatabricksEnvironment","value":"workerenv-7026851462233806"}]

    case class ClusterTags(key: String, value: String)
    implicit val formats = DefaultFormats
    try {
      val listOfClusterTags = parse(clusterTag)
      val clusterTagsMap = listOfClusterTags.extract[List[ClusterTags]].map(
        x => x.key -> x.value).toMap
      clusterTagsMap
    } catch {
      case NonFatal(_) =>
        logWarning(s"There was an exception parsing cluster tags string: $clusterTag, skipping")
        Map.empty
    }
  }

}
