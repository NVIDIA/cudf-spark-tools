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

package com.nvidia.spark.rapids.tool.tuning.plugins.delta

import com.nvidia.spark.rapids.tool.planparser.delta.DeltaLakeHelper
import com.nvidia.spark.rapids.tool.plugins.ConditionTrait
import com.nvidia.spark.rapids.tool.tuning.AutoTuner
import com.nvidia.spark.rapids.tool.tuning.plugins.{BaseTuningRule, TuningCondPredicates}
import org.apache.maven.artifact.versioning.ComparableVersion

/**
 * Delta Lake OSS Tuning Rules Module
 *
 * This module contains tuning rules specifically designed for optimizing Apache Spark applications
 * that use Delta Lake OSS (Open Source). These rules are part of the Delta Lake OSS Tuning Plugin
 * and provide automated recommendations for version compatibility, GPU acceleration features,
 * and migration guidance.
 *
 * ==Overview==
 * Delta Lake is an open-source storage framework that brings ACID transactions to Apache Spark
 * and big data workloads. The cuDF plugin provides limited support for
 * Delta Lake operations on GPU, with certain version requirements and configuration options.
 * This module helps users optimize their Delta Lake workloads for GPU acceleration.
 *
 * ==Rules Included==
 * This module defines three main tuning rules:
 *
 * 1. '''DeltaOSSVersionRule''': Validates Delta Lake version compatibility
 *    - Extracts the Delta Lake version from the application's classpath
 *    - Compares it against the minimum recommended version for the cuDF plugin
 *    - Generates recommendations if an upgrade is needed
 *    - Handles cases where Delta Lake version cannot be detected
 *
 * 2. '''DeltaOSSMigrationRule''': Provides migration guidance and support information
 *    - Informs users about Delta Lake support limitations in the cuDF plugin
 *    - Only activates for non-GPU enabled applications
 *    - Provides references to documentation for detailed feature support
 *
 * 3. '''DeltaOSSWriteGPURule''': Enables GPU-accelerated Delta Lake writes
 *    - Checks if GPU write acceleration is explicitly disabled
 *    - Recommends enabling the feature if not configured
 *    - Provides informational comments about default GPU write behavior
 *
 * ==Activation Conditions==
 * Each rule has specific conditions that determine when it should be applied:
 * - Version rule: Activates when Delta Lake version is below minimum or cannot be detected
 * - Migration rule: Activates only for applications not running on GPU
 * - Write rule: Activates when write acceleration is not explicitly configured
 *
 * ==Configuration Entries==
 * These rules reference the following tuning configuration entries:
 * - RECOMMENDED_DELTA_LAKE_VERSION: Minimum supported Delta Lake version
 * - DELTA_LAKE_OSS_VERSION_RECOMMENDATION: Comment template for version upgrades
 * - DELTA_LAKE_OSS_VERSION_UNDEFINED: Comment when version cannot be detected
 * - DELTA_LAKE_OSS_SUPPORT_COMMENT: General Delta Lake support information
 * - spark.rapids.sql.format.delta.write.enabled: GPU write acceleration property
 *
 * @see [[com.nvidia.spark.rapids.tool.tuning.plugins.delta.DeltaLakeOSSTuningPlugin]]
 * @see [[com.nvidia.spark.rapids.tool.planparser.delta.DeltaLakeHelper]]
 * @see [[com.nvidia.spark.rapids.tool.tuning.plugins.BaseTuningRule]]
 */

/**
 * Rule to check if the Delta Lake version used in the application
 * meets the minimum recommended version for optimal compatibility
 * with the cuDF plugin.
 */
class DeltaOSSVersionRule extends BaseTuningRule {
  private var extractedDLVersion: Option[String] = _
  private var minDeltaLakeVerRaw: String = _
  private var minDLVersion: ComparableVersion = _
  private var currentDLVersion: Option[ComparableVersion] = _

  private def getMinRecommendedDLVersion(tuner: AutoTuner): String = {
    tuner.configProvider.getEntry("RECOMMENDED_DELTA_LAKE_VERSION").min
  }

  override val condition: ConditionTrait[AutoTuner] = (tunerInst: AutoTuner) => {
    extractedDLVersion =
      DeltaLakeHelper.extractVersion(tunerInst.appInfoProvider.getClassPathEntries)
    minDeltaLakeVerRaw = getMinRecommendedDLVersion(tunerInst)
    minDLVersion = new ComparableVersion(minDeltaLakeVerRaw)

    extractedDLVersion match {
      case Some(version) =>
        val currentVersionObj = new ComparableVersion(version)
        currentDLVersion = Some(currentVersionObj)
        currentVersionObj.compareTo(minDLVersion) < 0
      case _ =>
        // could not find delta Lake version. Activate the rule anyway.
        true
    }
  }

  override def apply(tuner: AutoTuner): Unit = {
    var actualVersion = ""
    // pull the comment from the config based on whether we have a version or not.
    val commentKey = currentDLVersion match {
      case Some(v) =>
        actualVersion = v.toString
        "DELTA_LAKE_OSS_VERSION_RECOMMENDATION"
      case _ =>
        "DELTA_LAKE_OSS_VERSION_UNDEFINED"
    }
    val commentTemplate =
      tuner.configProvider.getEntry(commentKey).default
        .replace("[RECOMMENDED_VERSION]", minDLVersion.toString)
        .replace("[ACTUAL_VERSION]", actualVersion)
    tuner.appendComment(commentTemplate)
  }
}

/**
 * Rule to inform users about the limited support for Delta Lake tables
 * provided by the cuDF plugin when Delta Lake is detected in the application.
 */
class DeltaOSSMigrationRule extends BaseTuningRule {
  // get the related comment from the config file.
  private def getStaticComment(tuner: AutoTuner): String = {
    tuner.configProvider.getEntry("DELTA_LAKE_OSS_SUPPORT_COMMENT").default
  }

  // This rule only activated for non-cuDF-plugin event logs.
  override val condition: ConditionTrait[AutoTuner] = TuningCondPredicates.gpuNotEnabled

  override def apply(tuner: AutoTuner): Unit = {
    tuner.appendComment(getStaticComment(tuner))
  }
}

/**
 * Rule to enable GPU-accelerated writing for Delta Lake OSS
 * if the user has not explicitly disabled it.
 */
class DeltaOSSWriteGPURule extends BaseTuningRule {
  private val enableDeltaWriteProp = "spark.rapids.sql.format.delta.write.enabled"

  // Apply this rule if the user has not explicitly enforced the configuration.
  override val condition: ConditionTrait[AutoTuner] =
    TuningCondPredicates.nonEnforcedProperty(enableDeltaWriteProp)

  override def apply(tuner: AutoTuner): Unit = {
    tuner.getPropertyValue(enableDeltaWriteProp) match {
      case Some(value) if value.equalsIgnoreCase("false") =>
        tuner.appendRecommendation(enableDeltaWriteProp, "true")
      case _ =>
        // Add comment that the configuration is enabled by default on GPU.
        tuner.appendCommentFromLookup(enableDeltaWriteProp)
    }
  }
}
