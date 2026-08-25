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

package com.nvidia.spark.rapids.tool.tuning.plugins

import scala.util.control.NonFatal

import com.nvidia.spark.rapids.tool.tuning.AutoTuner
import com.nvidia.spark.rapids.tool.tuning.config.TuningPluginsConfig

import org.apache.spark.internal.Logging

/**
 * Manages the lifecycle and execution of tuning plugins for the cuDF plugin AutoTuner.
 *
 * The TuningPluginManager is responsible for:
 * - Loading and initializing tuning plugins from configuration
 * - Activating plugins based on their activation conditions
 * - Managing tuning rules associated with each plugin
 * - Coordinating the application of tuning rules across all active plugins
 * - Controlling the execution order of rules (per-plugin or globally sorted)
 *
 * This class follows a lazy initialization pattern where plugins are only built and activated
 * when first accessed. It uses the builder pattern to construct plugins dynamically based on
 * the plugin configuration.
 *
 * ==Rule Execution Order==
 * The manager supports two modes for applying tuning rules:
 * 1. '''Cross-plugin sorting''' (sortRulesAcrossPlugins = true, default):
 *    All rules from all plugins are collected and sorted by priority globally.
 *    This ensures strict priority ordering regardless of plugin boundaries.
 *
 * 2. '''Per-plugin sorting''' (sortRulesAcrossPlugins = false):
 *    Rules are executed plugin-by-plugin in plugin priority order, with rules
 *    within each plugin sorted by their own priorities. This preserves plugin
 *    grouping and can be useful for maintaining logical separation.
 *
 * @param tunerInst The AutoTuner instance that this manager will operate on. This instance
 *                  is passed to all plugins and rules for accessing tuning context and
 *                  applying configuration changes.
 * @param sortRulesAcrossPlugins If true, rules from all plugins are sorted globally by priority.
 *                               If false, plugins are processed in order and rules within each
 *                               plugin are sorted by priority. Default is true.
 *
 * @example
 * {{{
 *   val tuner = new AutoTuner(...)
 *
 *   // Default: rules sorted globally across all plugins
 *   val pluginManager = new TuningPluginManager(tuner, sortRulesAcrossPlugins = true)
 *   pluginManager.applyRules()
 *
 *   // Alternative: rules sorted per-plugin
 *   val perPluginManager = new TuningPluginManager(tuner, sortRulesAcrossPlugins = false)
 *   perPluginManager.applyRules()
 *
 *   // Get all tuning rules for inspection
 *   val rules = pluginManager.getTuningRulesSeq
 * }}}
 *
 * @see [[BaseTuningPlugin]] for plugin implementation details
 * @see [[com.nvidia.spark.rapids.tool.tuning.config.TuningPluginsConfig]] for configuration
 *      structure
 * @see [[AutoTuner]] for the tuner instance this manager operates on
 */
class TuningPluginManager(
    val tunerInst: AutoTuner,
    val sortRulesAcrossPlugins: Boolean,
    pluginsConf: TuningPluginsConfig) extends Logging {

  /**
   * Lazily initialized sequence of active plugins.
   *
   * Plugins are built from the default configuration and filtered based on their
   * activation conditions. Only plugins whose activation conditions evaluate to true
   * are included in this sequence.
   *
   * This is a lazy val to defer plugin instantiation and activation until actually needed,
   * improving startup performance.
   *
   * @return Sequence of activated tuning plugins, sorted by priority
   */
  private lazy val activePlugins: Seq[BaseTuningPlugin] = {
    buildPlugins(pluginsConf, tunerInst).filter(_.activate())
  }

  /**
   * Set of names of rules that have already been triggered. Note that a "triggered" rule does not
   * necessarily mean that it was applied as the rule's condition may not have been met.
   * This is used to prevent duplicate rule applications. It also can be used as a simple tracker
   * of which rules have been executed.
   */
  private val triggeredRules = scala.collection.mutable.Set[String]()

  /**
   * Builds tuning plugins from configuration.
   *
   * This method constructs plugin instances using the builder pattern:
   * 1. Filters enabled plugins from the configuration
   * 2. Sorts plugins by priority (0 = highest priority, 100 = lowest priority)
   * 3. Retrieves rules associated with each plugin from the plugin rules map
   * 4. For each plugin configuration with enabled rules:
   *    - Creates a builder instance
   *    - Configures the builder with plugin config, tuner instance, and associated rules
   *    - Builds the plugin instance
   *    - Handles build failures gracefully with warning logs
   *
   * Plugins without any enabled rules are skipped. Failed plugin instantiations are
   * logged and excluded from the result.
   *
   * @param config The tuning plugins configuration defining available plugins
   * @param tunerInst The AutoTuner instance to be passed to each plugin
   * @return Sequence of successfully built tuning plugins, sorted by priority
   */
  private def buildPlugins(
    config: TuningPluginsConfig, tunerInst: AutoTuner): Seq[BaseTuningPlugin] = {
    val pluginRulesMap = config.getPluginRulesMap
    config.getTuningPluginsSeq
      .filter(_.getEnabled)
      .sortBy(_.normalizedPriority)
      .flatMap { plugConf =>
        // get the rules associated with this plugin
        pluginRulesMap.getOrElse(plugConf.getName, Seq.empty).filter(_.enabled) match {
          case rulesSeq if rulesSeq.nonEmpty =>
            BaseTuningPlugin.getBuilder
              .withPluginConfig(plugConf)
              .withTuner(tunerInst)
              .addRules(rulesSeq)
              .build() match {
              case scala.util.Success(plugin) =>
                Some(plugin)
              case scala.util.Failure(e) =>
                logWarning(s"Failed to instantiate plugin ${plugConf.getName}: ${e.getMessage}")
                None
            }
          case _ =>
            // If rules are empty, then skip the entire plugin initialization
            logDebug("Plugin initialization skipped: " +
              s"${plugConf.getName} has no associated rules.")
            None
        }
    }
  }

  /**
   * Retrieves all tuning rules from active plugins, sorted by priority.
   *
   * Aggregates rules from all activated plugins and sorts them in ascending order
   * of priority (0 = highest priority, 100 = lowest priority). Rules with lower
   * priority values are executed first during rule application.
   *
   * This method is useful for:
   * - Inspecting which rules will be applied
   * - Understanding the order of rule execution
   * - Debugging rule conflicts or dependencies
   *
   * @return Sequence of tuning rules from all active plugins, sorted by priority
   *         (0 = highest, 100 = lowest)
   */
  def getTuningRulesSeq: Seq[TuningRuleTrait] = {
    activePlugins.flatMap(_.rules).sortBy(_.priority)
  }

  /**
   * Applies all tuning rules from active plugins to the AutoTuner instance.
   *
   * This method delegates to one of two strategies based on the `sortRulesAcrossPlugins`
   * configuration:
   * - If true: calls [[applyRulesAcrossPlugins]] to apply rules globally sorted by priority
   * - If false: calls [[applyRulesPerPlugin]] to apply rules grouped by plugin
   *
   * Each rule is only triggered once, tracked by the `triggeredRules` set to prevent
   * duplicate applications. Failed rule applications are logged as warnings but do not
   * stop the execution of subsequent rules.
   *
   * @see [[applyRulesAcrossPlugins]] for global rule sorting strategy
   * @see [[applyRulesPerPlugin]] for per-plugin rule sorting strategy
   */
  def applyRules(): Unit = {
    if (sortRulesAcrossPlugins) {
      applyRulesAcrossPlugins()
    } else {
      applyRulesPerPlugin()
    }
  }

  /**
   * Applies tuning rules grouped by plugin, maintaining plugin boundaries.
   *
   * This method processes active plugins in priority order, and for each plugin,
   * applies its rules sorted by their individual priorities. This approach preserves
   * logical grouping of rules within their parent plugins.
   *
   * '''Execution Flow:'''
   * 1. Iterate through plugins (already sorted by plugin priority)
   * 2. For each plugin, sort its rules by priority (0 = highest, 100 = lowest)
   * 3. Trigger each rule if not already triggered
   * 4. Track triggered rules to prevent duplicates
   * 5. Log warnings for failed rules but continue execution
   *
   * '''Use Case:'''
   * This strategy is useful when plugins have interdependent rules that should be
   * executed together, or when you want to maintain clear separation between
   * different tuning strategies implemented by different plugins.
   *
   * @note Failed rule applications are caught and logged but do not interrupt
   *       the processing of subsequent rules
   */
  protected def applyRulesPerPlugin(): Unit = {
    for (plugin <- activePlugins) {
      logDebug(s"Applying rules from plugin: ${plugin.getName}")
      for (rule <- plugin.rules.sortBy(_.priority)) {
        try {
          if (!triggeredRules.contains(rule.name)) {
            logDebug(s"Triggering rule: ${rule.name} with priority ${rule.priority}")
            rule.trigger(tunerInst)
            triggeredRules.add(rule.name)
          } else {
            logDebug(s"Skipping already triggered rule: ${rule.name}")
          }
        } catch {
          case NonFatal(e) =>
            logWarning(s"Failed to apply tuning rule ${rule.name}: ${e.getMessage}")
        }
      }
    }
  }

  /**
   * Applies tuning rules sorted globally across all plugins by priority.
   *
   * This method collects all rules from all active plugins and sorts them globally
   * by priority (0 = highest, 100 = lowest), regardless of which plugin they belong to.
   * This ensures strict priority ordering across the entire rule set.
   *
   * '''Execution Flow:'''
   * 1. Collect all rules from all active plugins via [[getTuningRulesSeq]]
   * 2. Sort rules globally by priority (0 = highest, 100 = lowest)
   * 3. Trigger each rule if not already triggered
   * 4. Track triggered rules to prevent duplicates
   * 5. Log warnings for failed rules but continue execution
   *
   * '''Use Case:'''
   * This strategy is useful when you need strict priority ordering regardless of
   * plugin boundaries. For example, if a high-priority rule from Plugin B should
   * execute before a low-priority rule from Plugin A, this method ensures that
   * ordering is respected.
   *
   * @note Failed rule applications are caught and logged but do not interrupt
   *       the processing of subsequent rules
   */
  protected def applyRulesAcrossPlugins(): Unit = {
    getTuningRulesSeq.foreach { r =>
      try {
        if (!triggeredRules.contains(r.name)) {
          r.trigger(tunerInst)
          triggeredRules.add(r.name)
        } else {
          logDebug("The rule " + r.name + " has already been triggered before, skipping...")
        }
      } catch {
        case e: Exception =>
          logWarning(s"Failed to apply tuning rule ${r.name}: ${e.getMessage}")
      }
    }
  }
}

object TuningPluginManager {
  class Builder {
    private var tunerInst: AutoTuner = _
    private var sortRulesAcrossPlugins: Boolean = true
    private var pluginConfigs: Option[TuningPluginsConfig] = None

    def withTunerInst(tuner: AutoTuner): Builder = {
      this.tunerInst = tuner
      this
    }

    def withSortRulesAcrossPlugins(sort: Boolean): Builder = {
      this.sortRulesAcrossPlugins = sort
      this
    }

    def withPluginsConfig(configs: TuningPluginsConfig): Builder = {
      this.pluginConfigs = Option(configs)
      this
    }

    def build(): TuningPluginManager = {
      if (pluginConfigs.isEmpty) {
        pluginConfigs = Some(tunerInst.configProvider.getPluginsConfigs)
      }
      new TuningPluginManager(
        tunerInst,
        sortRulesAcrossPlugins,
        pluginConfigs.getOrElse(new TuningPluginsConfig()))
    }
  }

  def builder: Builder = new Builder()
}
