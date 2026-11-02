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

import com.nvidia.spark.rapids.tool.plugins.{AppPropPlugConfig, AppPropRuntimeTrait, AppPropVersionExtractorTrait, BaseAppPropPlug, PropConditionTrait}

import org.apache.spark.sql.rapids.tool.util.SparkRuntime
import org.apache.spark.sql.rapids.tool.util.SparkRuntime.SparkRuntime


/**
 * Condition implementation for detecting Databricks runtime.
 * Identifies Databricks by checking cluster tag properties.
 */
class DBConditionImpl extends PropConditionTrait {
  // Cluster tags can be redacted, so we check ClusterId and ClusterName separately
  var clusterTags: String = ""
  var clusterTagClusterId: String = ""
  var clusterTagClusterName: String = ""

  /** Returns true if all required Databricks cluster properties are present. */
  private def isDatabricksInternal: Boolean = {
    clusterTags.nonEmpty && clusterTagClusterId.nonEmpty && clusterTagClusterName.nonEmpty
  }

  /**
   * Evaluates whether the application is running on Databricks by checking cluster tag properties.
   * @param props Application properties map.
   * @return true if Databricks is detected, false otherwise.
   */
  override def eval(props: collection.Map[String, String]): Boolean = {
    if (clusterTags.isEmpty) {
      clusterTags = props.getOrElse(DatabricksParseHelper.PROP_ALL_TAGS_KEY, "")
    }
    if (clusterTagClusterId.isEmpty) {
      clusterTagClusterId =
        props.getOrElse(DatabricksParseHelper.PROP_TAG_CLUSTER_ID_KEY, "")
    }
    if (clusterTagClusterName.isEmpty) {
      clusterTagClusterName =
        props.getOrElse(DatabricksParseHelper.PROP_TAG_CLUSTER_NAME_KEY, "")
    }
    isDatabricksInternal
  }
}

/**
 * Extended condition implementation that detects both Databricks and Photon runtime.
 * Photon is Databricks' vectorized execution engine.
 */
class DBPhotonConditionImpl extends DBConditionImpl {
  var isPhoton: Boolean = false
  var isDB: Boolean = false

  /**
   * Evaluates whether Photon is enabled in the application.
   * @param props Application properties map.
   * @return true if Photon is detected, false otherwise.
   */
  def evalPhoton(props: collection.Map[String, String]): Boolean = {
    isPhoton ||= PhotonParseHelper.eval(props)
    isPhoton
  }

  /**
   * Evaluates whether the application is running on Databricks and checks for Photon.
   * @param props Application properties map.
   * @return true if Databricks is detected (regardless of Photon), false otherwise.
   */
  override def eval(props: collection.Map[String, String]): Boolean = {
    isDB ||= super.eval(props)
    // we should reevaluate the photon anyway only if it is a databricks app
    if (isDB) {
      evalPhoton(props)
    }
    // return true if at least DB is true
    isDB
  }
}

/**
 * Trait for Databricks runtime detection that differentiates between
 * Photon and standard Databricks.
 */
trait DBPropRuntimeTrait extends AppPropRuntimeTrait {
  /** Returns true if Photon is enabled in the application. */
  def isPhotonEnabled: Boolean

  /**
   * Returns the detected runtime: PHOTON if enabled, None for standard Databricks.
   * @return Some(SparkRuntime.PHOTON) if Photon is enabled, None otherwise.
   */
  override def runtime: Option[SparkRuntime] = {
    if (isPhotonEnabled) {
      Some(SparkRuntime.PHOTON)
    } else {
      // TODO: currently we have no Databricks specific runtime. Consider it SPARK for now.
      None
    }
  }
}

/**
 * Plugin implementation for Databricks usage in Spark applications.
 * @param pluginConfig The configuration for the application property plugin.
 */
class DBPlugin(
    override val pluginConfig: AppPropPlugConfig
) extends BaseAppPropPlug(pluginConfig = pluginConfig)
  with DBPropRuntimeTrait {
  /** Condition evaluator for Databricks and Photon detection. */
  override val propCondition: DBPhotonConditionImpl = new DBPhotonConditionImpl

  /** Returns true if Photon is enabled based on condition evaluation. */
  override def isPhotonEnabled: Boolean = {
    propCondition match {
      case dbCond: DBPhotonConditionImpl => dbCond.isPhoton
      case _ => false
    }
  }

  /** Version extractor for Databricks runtime. */
  override val versionExtractor: Option[AppPropVersionExtractorTrait] = Some(DBVersionExtractor)

  /**
   * Post-enable action that extracts and stores the Databricks runtime version in plugin
   * metadata. It is not the Spark version, which the event log records.
   * @param properties Application properties map.
   */
  override def postEnableAction(properties: collection.Map[String, String]): Unit = {
    // set the runtime version if possible using the version extractor from the spark properties.
    evalPluginVersion(properties) match {
      case Some(version) => addMetadata("runtime.version", version)
      case None => // do nothing
    }
  }

  // --- Backward Compatibility Methods ---
  // The following methods provide backward compatibility when AppBase used to extend
  // the ClusterTagPropHandler trait. They delegate to the propCondition instance.

  /** Returns the raw cluster tags string. */
  def clusterTags: String = propCondition.clusterTags

  /** Returns the cluster ID extracted from properties. */
  def clusterTagClusterId: String = propCondition.clusterTagClusterId

  /** Returns the cluster name extracted from properties. */
  def clusterTagClusterName: String = propCondition.clusterTagClusterName

  /**
   * Prepares a map of cluster tags by combining various cluster property sources.
   * Extracts cluster ID and job ID from available properties.
   * @return Map of cluster tag key-value pairs.
   */
  def prepareClusterTags: Map[String, String] = {
    // prepare the cluster tags map
    val initialClusterTagsMap = if (clusterTags.nonEmpty) {
      DatabricksParseHelper.parseClusterTags(clusterTags)
    } else {
      Map.empty[String, String]
    }
    val tagsMapWithClusterId =
      if (!initialClusterTagsMap.contains(DatabricksParseHelper.SUB_PROP_CLUSTER_ID)
        && clusterTagClusterId.nonEmpty) {
        initialClusterTagsMap + (DatabricksParseHelper.SUB_PROP_CLUSTER_ID -> clusterTagClusterId)
      } else {
        initialClusterTagsMap
      }

    if(!tagsMapWithClusterId.contains(DatabricksParseHelper.SUB_PROP_JOB_ID)
      && clusterTagClusterName.nonEmpty) {
      val clusterTagJobId = DatabricksParseHelper.parseClusterNameForJobId(clusterTagClusterName)
      clusterTagJobId.map { jobId =>
        tagsMapWithClusterId + (DatabricksParseHelper.SUB_PROP_JOB_ID -> jobId)
      }.getOrElse(tagsMapWithClusterId)
    } else {
      tagsMapWithClusterId
    }
  }
}
