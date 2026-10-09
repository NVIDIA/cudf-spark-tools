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
package com.nvidia.spark.rapids.tool.planparser.auron

import java.nio.file.Paths

import scala.util.matching.Regex

import com.nvidia.spark.rapids.tool.plugins.{AppPropVersionExtractorFromCPTrait, ConditionTrait, OssOpMapperFromFileTrait, PropConditionOnSparkExtTrait}

import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.util.stubs.SparkPlanExtensions.UpStreamSparkPlanInfoOps
import org.apache.spark.sql.rapids.tool.util.stubs.SparkPlanInfo
import org.apache.spark.sql.rapids.tool.util.stubs.auron.AuronSparkPlanInfo

/**
 * Mapper for Auron (Native Spark) execution operators.
 *
 * Auron is an accelerated Spark execution engine that uses native operators with "Native" prefix
 * (e.g., NativeProject, NativeFilter) and converter operators (ConvertToNative). This mapper
 * transforms Auron operators to OSS Spark equivalents and handles Auron-specific terminology.
 *
 * Key differences from parent:
 * - Loads operator mappings from JSON file (auron-oss-op-map-6_0.json)
 * - Handles special "ConvertToNative" operator in addition to Native* pattern
 * - Translates Auron join terminology: BroadcastLeft/BroadcastRight → BuildLeft/BuildRight
 * - Only active when Auron extension is enabled in application
 *
 * Example transformations:
 * - NativeProject → Project
 * - NativeHashAggregate → HashAggregate
 * - ConvertToNative → (mapped via JSON)
 * - Join descriptions: "BroadcastLeft" → "BuildLeft"
 */
object AuronOssOpMapper extends OssOpMapperFromFileTrait {
  /** Pattern matching Auron operators: "Native*" or "ConvertToNative" */
  private val AURON_PATTERN: Regex = "Native[A-Z][a-zA-Z]+|ConvertToNative".r

  /** Directory containing Auron operator mapping files */
  private val OPS_MAPPING_DIR = "parser/auron"

  /** Default operator mapping file for Auron version 6.0 */
  private val DEFAULT_OPS_MAPPING_FILE = "auron-oss-op-map-6_0.json"

  /**
   * Mapping for Auron-specific join terminology in node descriptions.
   * Auron native converters use BroadcastLeft/BroadcastRight instead of OSS BuildLeft/BuildRight.
   * Note that this is expected to grow up as we identify more Auron-specific terms.
   */
  private val descKeywordsMapping: Map[String, String] = Map(
    "BroadcastLeft" -> "BuildLeft",
    "BroadcastRight" -> "BuildRight"
  )

  /** Checks if a node name matches the Auron operator pattern */
  private def isAuronNode(nodeName: String): Boolean = {
    AURON_PATTERN.findFirstIn(nodeName).isDefined
  }

  override def mappingFilePath: String =
    Paths.get(OPS_MAPPING_DIR, DEFAULT_OPS_MAPPING_FILE).toString

  /** Only active when Auron is enabled in the application */
  override val appCtxtCondition: ConditionTrait[AppBase] = (app: AppBase) => app.isAuronEnabled

  override def acceptPlanInfo(planInfo: org.apache.spark.sql.execution.SparkPlanInfo): Boolean = {
    isAuronNode(planInfo.nodeName)
  }

  /**
   * Maps Auron plan data to OSS equivalents with additional keyword translation.
   *
   * Unlike the parent implementation, this also translates Auron-specific join terminology
   * in the description (BroadcastLeft/BroadcastRight → BuildLeft/BuildRight).
   */
  override def mapPlanDataToOss(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo
  ): (String, String) = {
    var modifiedDesc = mapContentToOss(planInfo.simpleString)

    // Apply descMapping replacements to convert Auron keywords to OSS equivalents
    descKeywordsMapping.foreach { case (auronKeyword, ossKeyword) =>
      modifiedDesc = modifiedDesc.replace(auronKeyword, ossKeyword)
    }

    (mapContentToOss(planInfo.nodeName), modifiedDesc)
  }

  /**
   * Creates an AuronSparkPlanInfo preserving both Auron and OSS representations.
   * Uses AuronSparkPlanInfo instead of generic SparkPlanInfo to maintain Auron context.
   */
  override def toPlatformAwarePlan(
      planInfo: org.apache.spark.sql.execution.SparkPlanInfo,
      app: AppBase): SparkPlanInfo = {
    val (ossPlanName, ossPlanDesc) = mapPlanDataToOss(planInfo)
    AuronSparkPlanInfo(
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
   * Maps Auron operator names to OSS equivalents using the loaded JSON mapping.
   * Performs lookup-based transformation (e.g., NativeProject → Project).
   */
  override def mapContentToOss(inputStr: String): String = {
    AURON_PATTERN.replaceAllIn(inputStr, m => ossSparkMapping.getOrElse(m.matched, m.matched))
  }
}

/**
 * Helper object for detecting and validating Auron-enabled Spark applications.
 *
 * Auron is identified by the presence of AuronSparkSessionExtension in spark.sql.extensions
 * and the spark.auron.enabled property. Also extracts Auron version from classpath JAR names.
 *
 * Detection criteria:
 * - spark.sql.extensions contains "AuronSparkSessionExtension"
 * - spark.auron.enabled is "true" (default if not set)
 * - Classpath contains auron-spark-*.jar with version pattern
 */
object AuronParseHelper extends PropConditionOnSparkExtTrait
    with AppPropVersionExtractorFromCPTrait {
  /** Property key controlling Auron execution */
  private val AURON_ENABLED_KEY = "spark.auron.enabled"

  /** Default value when spark.auron.enabled is not explicitly set */
  private val AURON_ENABLED_KEY_DEFAULT = "true"

  /** Extension patterns identifying Auron-enabled applications */
  override val extensionRegxMap: Map[String, String] = Map(
    "spark.sql.extensions" -> ".*AuronSparkSessionExtension.*"
  )

  /**
   * Pattern to extract Auron version from classpath JAR name.
   * Example: auron-spark-3.5-6.0.0-incubating.jar
   */
  override val cpKeyRegex: Regex = """auron-spark-[\w.-]+-(\d+\.\d+\.\d+)-incubating\.jar""".r

  /**
   * Evaluates if Auron is enabled by checking both extension presence and spark.auron.enabled flag.
   * Unlike parent which only checks extension, this also validates the enabled property.
   */
  override def eval(properties: collection.Map[String, String]): Boolean = {
    val auronExtensionExists = super.eval(properties)
    auronExtensionExists && isAuronTurnedOn(properties)
  }

  /** Checks if spark.auron.enabled is set to "true" (case-insensitive) */
  private def isAuronTurnedOn(properties: collection.Map[String, String]): Boolean = {
    properties.getOrElse(AURON_ENABLED_KEY, AURON_ENABLED_KEY_DEFAULT)
      .trim.equalsIgnoreCase(AURON_ENABLED_KEY_DEFAULT)
  }
}
