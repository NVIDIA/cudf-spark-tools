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

import com.nvidia.spark.rapids.tool.planparser.db.DBPlugin

import org.apache.spark.sql.rapids.tool.util.SparkRuntime.SparkRuntime

/**
 * Trait that provides functionality to manage and interact with application property plugins.
 * It initializes a map of available plugins, allows re-evaluation of plugins based on
 * application properties, and provides methods to check if specific plugins are enabled.
 */
trait AppPropPlugContainerTrait {
  // Initialize the plugin map once
  def initPluginMap(): Map[String, BaseAppPropPlug] = BaseAppPropPlug.loadAppPropPlugs()

  // Map of plugin ID to AppPropPlugTrait instances
  val pluginMap: Map[String, BaseAppPropPlug] = initPluginMap()

  /**
   * Retrieves a plugin instance by its ID with type-safe casting.
   * @tparam T The type of plugin to return, must extend BaseAppPropPlug
   * @param pluginId The ID of the plugin to retrieve
   * @return The plugin instance cast to the requested type
   */
  @throws[ClassCastException]
  @throws[NoSuchElementException]
  def getPluginInst[T <: BaseAppPropPlug](pluginId: String): T = {
    pluginMap(pluginId).asInstanceOf[T]
  }

  /**
   * Re-evaluates all plugins with the provided application properties. This is typically invoked
   * when the application properties change, allowing each plugin to update its state accordingly.
   * For example, during an environment update event, or job start.
   * @param properties Application properties to re-evaluate the plugins against.
   */
  def reEvaluate(properties: collection.Map[String, String]): Unit = {
    pluginMap.values.foreach(_.reEvaluate(properties))
  }

  def reEvaluateOnJobLevel(jobProperties: collection.Map[String, String]): Unit = {
    pluginMap.values
      .filter(_.hasJobLevelConfigs)
      .foreach(_.reEvaluate(jobProperties))
  }

  /**
   * Checks if a plugin with the given ID is enabled.
   * @param pluginId The ID of the plugin to check.
   * @return true if the plugin is enabled, false otherwise.
   */
  def isEnabled(pluginId: String): Boolean = {
    pluginMap.get(pluginId).exists(_.isEnabled)
  }

  /**
   * Retrieves the Spark runtime associated with the enabled plugins.
   * @return An Option containing the SparkRuntime if any enabled plugin has a defined runtime,
   *         otherwise None.
   */
  def getSparkRuntimePlugins: Option[SparkRuntime] = {
    pluginMap.values.filter(_.isEnabled).find(_.runtime.isDefined).flatMap(_.runtime)
  }

  // Begin definition of shortcuts for plugin enabled checks.

  // A flag to indicate whether the spark App is configured to use Auron.
  def isAuronEnabled: Boolean = {
    isEnabled(BaseAppPropPlug.AURON_PLUG_ID)
  }
  // A flag to indicate whether the spark App is configured to use DeltaLake.
  // Note that this is only a best-effort flag based on the spark properties.
  def isDeltaLakeOSSEnabled: Boolean = {
    isEnabled(BaseAppPropPlug.DELTA_OSS_PLUG_ID)
  }
  // A flag to indicate whether the spark App is configured to use Iceberg.
  // Note that this is only a best-effort flag based on the spark properties.
  def isIcebergEnabled: Boolean = {
    isEnabled(BaseAppPropPlug.ICEBERG_PLUG_ID)
  }
  def isHiveEnabled: Boolean = {
    isEnabled(BaseAppPropPlug.HIVE_PLUG_ID)
  }

  def dbPlugin: DBPlugin = {
    getPluginInst[DBPlugin](BaseAppPropPlug.DB_PLUG_ID)
  }
}
